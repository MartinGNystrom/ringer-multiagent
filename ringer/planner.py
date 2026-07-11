"""The planner (docs/design.html §4/§5): plans once, never does the work.

The planner sees only lightweight previews of each unit -- never the full
corpus -- so its own context stays small regardless of how big the pile is.
That's the harness's answer to the context/memory hard limit (§2): the
planner scales with the *number* of units, not their combined size.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._client import call_json
from .models import Role, Tier, resolve_model
from .spec import TaskSpec

_PLANNER_SYSTEM = """\
You are the planning stage of a multi-agent harness. You write specs; you \
never do the work yourself. For each unit in the manifest, decide:

- clear, self-contained instructions a worker with no other context can \
  follow to produce output matching the given output schema
- whether this unit's correctness is subjective enough that a fresh-eyes \
  reviewer must check it before it's trusted (needs_judge), and if so, a \
  short rubric for that reviewer
- whether the unit is routine enough for a cheaper "thrift" worker tier, \
  or needs the "default" tier

Write one spec per unit_id in the manifest. Do not invent unit_ids. Do not \
attempt the task yourself -- only produce specs."""

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "specs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string"},
                    "instructions": {"type": "string"},
                    "tier": {"type": "string", "enum": ["default", "thrift"]},
                    "needs_judge": {"type": "boolean"},
                    "judge_rubric": {"type": ["string", "null"]},
                },
                "required": ["unit_id", "instructions", "tier", "needs_judge", "judge_rubric"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["specs"],
    "additionalProperties": False,
}


@dataclass
class PlanResult:
    specs: list[TaskSpec]
    input_tokens: int
    output_tokens: int
    cost: float
    model: str


async def plan(
    *,
    task_description: str,
    output_schema: dict,
    units: list[dict],
    tier: Tier = Tier.DEFAULT,
    max_tokens: int = 8000,
    effort: str = "high",
) -> PlanResult:
    """Decompose a task into one TaskSpec per unit.

    `units` is a list of {"unit_id": str, "preview": str} -- previews only,
    never full content (see module docstring). The caller attaches the real
    `input_data` afterward by unit_id; see orchestrator.py.
    """
    model = resolve_model(Role.PLANNER, tier)

    manifest = "\n".join(f"- {u['unit_id']}: {u.get('preview', '')[:400]}" for u in units)
    user_content = (
        f"TASK\n{task_description}\n\n"
        f"OUTPUT SCHEMA EACH WORKER MUST SATISFY\n{output_schema}\n\n"
        f"UNIT MANIFEST ({len(units)} units)\n{manifest}"
    )

    result = await call_json(
        model=model,
        system=_PLANNER_SYSTEM,
        user_content=user_content,
        schema=_PLAN_SCHEMA,
        max_tokens=max_tokens,
        effort=effort,
    )

    known_ids = {u["unit_id"] for u in units}
    specs: list[TaskSpec] = []
    for raw in result.parsed["specs"]:
        if raw["unit_id"] not in known_ids:
            # The planner isn't allowed to invent units it wasn't given.
            continue
        specs.append(
            TaskSpec(
                unit_id=raw["unit_id"],
                instructions=raw["instructions"],
                input_data=None,  # filled in by the orchestrator from `units`
                output_schema=output_schema,
                tier=raw["tier"],
                needs_judge=raw["needs_judge"],
                judge_rubric=raw.get("judge_rubric"),
            )
        )

    # Any unit the planner silently dropped still needs a spec -- fail open
    # with a generic instruction rather than losing the unit entirely.
    covered = {s.unit_id for s in specs}
    for u in units:
        if u["unit_id"] not in covered:
            specs.append(
                TaskSpec(
                    unit_id=u["unit_id"],
                    instructions=task_description,
                    input_data=None,
                    output_schema=output_schema,
                    tier="default",
                    needs_judge=False,
                )
            )

    return PlanResult(
        specs=specs,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost=result.cost,
        model=result.served_by,
    )
