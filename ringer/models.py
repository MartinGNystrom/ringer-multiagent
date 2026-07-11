"""Model IDs, pricing, providers, and role -> model resolution.

This is the one file the design doc (docs/design.html §6) treats as the
source of truth at runtime. If Anthropic ships a new model, or OpenRouter's
catalog changes, update it here and every role resolves it automatically.

Two providers are wired in. The planner and judge always stay on Anthropic
-- that's the part of the Ringer thesis that doesn't change: a strong,
trusted model plans and grades. Workers can route to either Anthropic
(Sonnet 5 / Haiku 4.5) or, for the cheapest high-volume units, open-weight
models served through OpenRouter (GLM 5.2, Kimi K2). The mechanical
checker (checker.py) is what makes routing untrusted-provider workers safe
at all -- see docs/design.html §2 on the validation limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class Role(str, Enum):
    PLANNER = "planner"
    WORKER = "worker"
    JUDGE = "judge"


class Tier(str, Enum):
    """Cost/capability tier within a role.

    DEFAULT is the everyday choice for that role. ESCALATED is reserved for
    the harder cases the docs describe: a genuinely novel task shape for the
    planner, or a high-stakes unit (compliance, legal, financial) for the
    judge. Workers have three tiers below default instead of ESCALATED:
    THRIFT (Haiku -- pure volume, low-complexity Anthropic units) and two
    OpenRouter tiers for open-weight models, cheaper still, at the cost of
    less reliable structured output (the harness's retry loop is what makes
    that an acceptable trade -- see worker.py and _openrouter_client.py).
    """

    DEFAULT = "default"
    ESCALATED = "escalated"
    THRIFT = "thrift"
    OPENROUTER_GLM = "openrouter_glm"
    OPENROUTER_KIMI = "openrouter_kimi"


MODEL_FABLE = "claude-fable-5"
MODEL_OPUS = "claude-opus-4-8"
MODEL_SONNET = "claude-sonnet-5"
MODEL_HAIKU = "claude-haiku-4-5"

# OpenRouter model slugs -- fetched live from openrouter.ai/api/v1/models
# rather than guessed; re-check that endpoint if these ever 404.
MODEL_GLM_5_2 = "z-ai/glm-5.2"
MODEL_KIMI_K2 = "moonshotai/kimi-k2.7-code"

ANTHROPIC_MODELS = {MODEL_FABLE, MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU}
OPENROUTER_MODELS = {MODEL_GLM_5_2, MODEL_KIMI_K2}


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_per_mtok
            + output_tokens / 1_000_000 * self.output_per_mtok
        )


# Pricing as documented in docs/design.html §3/§6. Sonnet 5 intro pricing
# runs through 2026-08-31; update to (3.00, 15.00) after that if still
# current. GLM/Kimi pricing pulled live from OpenRouter's own model catalog.
PRICING: dict[str, ModelPricing] = {
    MODEL_FABLE: ModelPricing(10.00, 50.00),
    MODEL_OPUS: ModelPricing(5.00, 25.00),
    MODEL_SONNET: ModelPricing(2.00, 10.00),
    MODEL_HAIKU: ModelPricing(1.00, 5.00),
    MODEL_GLM_5_2: ModelPricing(0.35, 1.10),
    MODEL_KIMI_K2: ModelPricing(0.72, 3.50),
}

# Role/tier -> model. This is the literal implementation of the §6 table.
_RESOLUTION: dict[tuple[Role, Tier], str] = {
    (Role.PLANNER, Tier.DEFAULT): MODEL_OPUS,
    (Role.PLANNER, Tier.ESCALATED): MODEL_FABLE,
    (Role.WORKER, Tier.DEFAULT): MODEL_SONNET,
    (Role.WORKER, Tier.THRIFT): MODEL_HAIKU,
    (Role.WORKER, Tier.OPENROUTER_GLM): MODEL_GLM_5_2,
    (Role.WORKER, Tier.OPENROUTER_KIMI): MODEL_KIMI_K2,
    (Role.JUDGE, Tier.DEFAULT): MODEL_OPUS,
    (Role.JUDGE, Tier.ESCALATED): MODEL_FABLE,
}


def resolve_model(role: Role, tier: Tier = Tier.DEFAULT) -> str:
    """Resolve a (role, tier) pair to a concrete model ID.

    Raises KeyError with a helpful message for invalid combinations (e.g.
    Role.JUDGE + Tier.OPENROUTER_GLM -- the judge never leaves Anthropic).
    """
    try:
        return _RESOLUTION[(role, tier)]
    except KeyError as exc:
        valid = sorted(f"{r.value}/{t.value}" for r, t in _RESOLUTION)
        raise KeyError(
            f"No model mapped for {role.value}/{tier.value}. Valid combinations: {valid}"
        ) from exc


def provider_of(model: str) -> Provider:
    if model in OPENROUTER_MODELS:
        return Provider.OPENROUTER
    return Provider.ANTHROPIC


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0
    return pricing.cost(input_tokens, output_tokens)
