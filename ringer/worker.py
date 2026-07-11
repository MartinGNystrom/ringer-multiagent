"""Workers (docs/design.html §4/§5): burn most of the tokens, get no trust.

A worker executes exactly one TaskSpec and returns output matching the
spec's schema. Its own confidence is never read by anything downstream --
the checker and judge decide, not the worker's tone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._client import call_json
from .models import Role, Tier, resolve_model
from .spec import TaskSpec

_WORKER_SYSTEM = """\
You execute one unit of work against a spec written by a planning stage you \
cannot see or query. Follow the instructions exactly. Produce output that \
validates against the given schema. If the instructions ask you to cite \
sources for claims, every claim must name a real source from the input --\
never invent one. If a previous attempt at this unit was rejected, the \
rejection reason is included below -- fix that specific problem; do not \
just retry the same output."""


@dataclass
class WorkResult:
    output: Any
    input_tokens: int
    output_tokens: int
    cost: float
    model: str


async def run(spec: TaskSpec, *, max_tokens: int = 8000, effort: str = "medium") -> WorkResult:
    tier = Tier.THRIFT if spec.tier == "thrift" else Tier.DEFAULT
    model = resolve_model(Role.WORKER, tier)

    if spec.output_schema is None:
        raise ValueError(f"unit {spec.unit_id!r} has no output_schema; the planner or task author must set one")

    parts = [f"INSTRUCTIONS\n{spec.instructions}", f"INPUT\n{spec.input_data}"]
    if spec.failure_context:
        history = "\n".join(f"- attempt {i + 1} rejected: {reason}" for i, reason in enumerate(spec.failure_context))
        parts.append(f"PRIOR REJECTIONS -- FIX THESE\n{history}")
    user_content = "\n\n".join(parts)

    result = await call_json(
        model=model,
        system=_WORKER_SYSTEM,
        user_content=user_content,
        schema=spec.output_schema,
        max_tokens=max_tokens,
        effort=effort,
    )

    return WorkResult(
        output=result.parsed,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost=result.cost,
        model=result.served_by,
    )
