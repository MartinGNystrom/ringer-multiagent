"""The judge (docs/design.html §4/§5): fresh eyes, rationed, never the author.

The judge sees only the spec and the worker's output -- never which worker
produced it, never prior attempts, never the worker's own confidence. It's
only invoked for checker-passed results that are still subjective (spec
.needs_judge or checker.CheckResult.needs_judge) -- everything with a
mechanical answer never reaches this module at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._client import call_json
from .models import Role, Tier, resolve_model
from .spec import TaskSpec

_JUDGE_SYSTEM = """\
You are an independent reviewer. You did not write the output you are \
reviewing and have no information about who or what produced it -- judge it \
purely on whether it satisfies the spec below. Do not give the benefit of \
the doubt for effort, confidence, or plausible-sounding language: if the \
rubric or instructions are not clearly satisfied, reject it. Be specific in \
your reason so a worker could fix the exact problem on the next attempt."""

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["passed", "reason"],
    "additionalProperties": False,
}


@dataclass
class JudgeResult:
    passed: bool
    reason: str
    input_tokens: int
    output_tokens: int
    cost: float
    model: str


async def evaluate(
    spec: TaskSpec,
    output: Any,
    *,
    tier: Tier = Tier.DEFAULT,
    max_tokens: int = 4000,
    effort: str = "high",
) -> JudgeResult:
    model = resolve_model(Role.JUDGE, tier)

    rubric = spec.judge_rubric or "Judge strictly against the instructions below -- no separate rubric was given."
    user_content = (
        f"INSTRUCTIONS THE OUTPUT WAS SUPPOSED TO SATISFY\n{spec.instructions}\n\n"
        f"RUBRIC\n{rubric}\n\n"
        f"OUTPUT UNDER REVIEW\n{output}"
    )

    result = await call_json(
        model=model,
        system=_JUDGE_SYSTEM,
        user_content=user_content,
        schema=_VERDICT_SCHEMA,
        max_tokens=max_tokens,
        effort=effort,
    )

    return JudgeResult(
        passed=bool(result.parsed["passed"]),
        reason=result.parsed["reason"],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost=result.cost,
        model=result.served_by,
    )
