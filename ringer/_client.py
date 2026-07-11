"""Shared Anthropic call plumbing used by planner/worker/judge.

Centralizing this in one place is what makes the per-model rules in
docs/design.html §6 actually enforced rather than merely documented: adaptive
thinking is only sent to models that support it, effort is only sent where
it won't 400, and Claude Fable 5 always goes through the beta endpoint with
a server-side fallback to Opus 4.8 (per the Fable migration guide -- a
refused request should not just silently stop).
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from .models import MODEL_FABLE, MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET, estimate_cost

_async_client: anthropic.AsyncAnthropic | None = None


def get_async_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic()
    return _async_client


def _thinking_config(model: str) -> dict | None:
    """Adaptive thinking where supported; omitted where it would 400 or is a no-op.

    Opus 4.8 / Sonnet 5 need it set explicitly to turn thinking *on*.
    Fable 5 has thinking always on -- explicitly setting {"type":"disabled"}
    is what 400s there, so we simply omit the param. Haiku 4.5 doesn't
    support adaptive thinking at all.
    """
    if model in (MODEL_OPUS, MODEL_SONNET):
        return {"type": "adaptive"}
    return None


def _supports_effort(model: str) -> bool:
    # Effort errors on Haiku 4.5; Fable 5 / Opus 4.8 / Sonnet 5 all support it.
    return model in (MODEL_FABLE, MODEL_OPUS, MODEL_SONNET)


class RefusalError(RuntimeError):
    """Raised when the model -- or the whole Fable fallback chain -- refuses."""


class CallResult:
    __slots__ = ("parsed", "input_tokens", "output_tokens", "cost", "served_by")

    def __init__(self, parsed: Any, input_tokens: int, output_tokens: int, cost: float, served_by: str):
        self.parsed = parsed
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost = cost
        self.served_by = served_by


def _extract_text(content) -> str:
    for block in content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("response contained no text content block to parse as JSON")


async def call_json(
    *,
    model: str,
    system: str,
    user_content: str,
    schema: dict,
    max_tokens: int = 8000,
    effort: str = "high",
) -> CallResult:
    """Call `model`, constraining the response to `schema`, and parse it.

    Structured outputs (`output_config.format`) guarantee the first text
    block is valid JSON matching the schema, so no retry-on-parse-failure
    logic is needed here -- that's what the checker (checker.py) is for.
    """
    client = get_async_client()

    output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
    if _supports_effort(model):
        output_config["effort"] = effort

    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        output_config=output_config,
        messages=[{"role": "user", "content": user_content}],
    )
    thinking = _thinking_config(model)
    if thinking is not None:
        kwargs["thinking"] = thinking

    if model == MODEL_FABLE:
        response = await client.beta.messages.create(
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": MODEL_OPUS}],
            **kwargs,
        )
    else:
        response = await client.messages.create(**kwargs)

    if response.stop_reason == "refusal":
        raise RefusalError(f"model {model} (and any configured fallback) refused the request")

    text = _extract_text(response.content)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model {model} returned non-JSON despite output_config.format: {text[:300]!r}") from exc

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    served_by = response.model
    cost = estimate_cost(served_by, input_tokens, output_tokens)
    return CallResult(parsed, input_tokens, output_tokens, cost, served_by)
