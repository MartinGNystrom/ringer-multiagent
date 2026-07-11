"""Model IDs, pricing, and role -> model resolution.

This is the one file the design doc (docs/design.html §6) treats as the
source of truth at runtime. If Anthropic ships a new model, update it here
and every role resolves it automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    PLANNER = "planner"
    WORKER = "worker"
    JUDGE = "judge"


class Tier(str, Enum):
    """Cost/capability tier within a role.

    DEFAULT is the everyday choice for that role. ESCALATED is reserved for
    the harder cases the docs describe: a genuinely novel task shape for the
    planner, or a high-stakes unit (compliance, legal, financial) for the
    judge. Workers have a THRIFT tier instead: pure volume, low-complexity
    units where even Sonnet is overkill.
    """

    DEFAULT = "default"
    ESCALATED = "escalated"
    THRIFT = "thrift"


MODEL_FABLE = "claude-fable-5"
MODEL_OPUS = "claude-opus-4-8"
MODEL_SONNET = "claude-sonnet-5"
MODEL_HAIKU = "claude-haiku-4-5"


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_per_mtok
            + output_tokens / 1_000_000 * self.output_per_mtok
        )


# Pricing as documented in docs/design.html §3. Sonnet 5 intro pricing runs
# through 2026-08-31; update to (3.00, 15.00) after that if still current.
PRICING: dict[str, ModelPricing] = {
    MODEL_FABLE: ModelPricing(10.00, 50.00),
    MODEL_OPUS: ModelPricing(5.00, 25.00),
    MODEL_SONNET: ModelPricing(2.00, 10.00),
    MODEL_HAIKU: ModelPricing(1.00, 5.00),
}

# Role/tier -> model. This is the literal implementation of the §6 table.
_RESOLUTION: dict[tuple[Role, Tier], str] = {
    (Role.PLANNER, Tier.DEFAULT): MODEL_OPUS,
    (Role.PLANNER, Tier.ESCALATED): MODEL_FABLE,
    (Role.WORKER, Tier.DEFAULT): MODEL_SONNET,
    (Role.WORKER, Tier.THRIFT): MODEL_HAIKU,
    (Role.JUDGE, Tier.DEFAULT): MODEL_OPUS,
    (Role.JUDGE, Tier.ESCALATED): MODEL_FABLE,
}


def resolve_model(role: Role, tier: Tier = Tier.DEFAULT) -> str:
    """Resolve a (role, tier) pair to a concrete model ID.

    Raises KeyError with a helpful message for invalid combinations (e.g.
    Role.WORKER + Tier.ESCALATED, which isn't a thing workers have).
    """
    try:
        return _RESOLUTION[(role, tier)]
    except KeyError as exc:
        valid = sorted(f"{r.value}/{t.value}" for r, t in _RESOLUTION)
        raise KeyError(
            f"No model mapped for {role.value}/{tier.value}. Valid combinations: {valid}"
        ) from exc


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0
    return pricing.cost(input_tokens, output_tokens)
