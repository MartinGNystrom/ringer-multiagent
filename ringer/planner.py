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
- which worker tier the unit needs (see the tier guidance below)

Write one spec per unit_id in the manifest. Do not invent unit_ids. Do not \
attempt the task yourself -- only produce specs."""

_TIER_GUIDANCE_CLOSED = """\
- "default": the everyday tier for this unit
- "thrift": routine, low-complexity units (simple lookups, short classification) \
  where a cheaper model is enough"""

_TIER_GUIDANCE_OPEN = """\
- "default": the everyday tier for this unit
- "thrift": routine, low-complexity Anthropic-tier work
- "openrouter_glm" / "openrouter_deepseek": general-purpose open-weight \
  models routed through OpenRouter, cheaper still than "thrift" -- use these \
  for the highest-volume, lowest-stakes units where cost matters most and \
  the mechanical checker can catch a bad output on its own
- "openrouter_kimi" / "openrouter_qwen_coder": open-weight models tuned for \
  code -- prefer these over the general-purpose OpenRouter tiers when the \
  unit itself is a coding task (writing/reviewing code, not prose or \
  extraction)

All four OpenRouter tiers have lower structured-output reliability than an \
Anthropic model, so avoid them for units the checker can't fully verify \
(i.e. anything you'd also mark needs_judge)."""


def _plan_schema(allow_openrouter: bool) -> dict:
    tiers = ["default", "thrift"]
    if allow_openrouter:
        tiers += ["openrouter_glm", "openrouter_kimi", "openrouter_deepseek", "openrouter_qwen_coder"]
    return {
        "type": "object",
        "properties": {
            "specs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "unit_id": {"type": "string"},
                        "instructions": {"type": "string"},
                        "tier": {"type": "string", "enum": tiers},
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
    allow_openrouter: bool = False,
    max_tokens: int = 8000,
    effort: str = "high",
) -> PlanResult:
    """Decompose a task into one TaskSpec per unit.

    `units` is a list of {"unit_id": str, "preview": str} -- previews only,
    never full content (see module docstring). The caller attaches the real
    `input_data` afterward by unit_id; see orchestrator.py.

    `allow_openrouter` opts the planner into routing units to the
    OpenRouter worker tiers (models.py Tier.OPENROUTER_GLM/_KIMI/_DEEPSEEK/
    _QWEN_CODER). It's off by default -- a task author has to explicitly
    decide their units can tolerate a less reliable, third-party
    structured-output path before the planner is even allowed to reach
    for it.
    """
    model = resolve_model(Role.PLANNER, tier)
    tier_guidance = _TIER_GUIDANCE_OPEN if allow_openrouter else _TIER_GUIDANCE_CLOSED
    system = f"{_PLANNER_SYSTEM}\n\nTIER OPTIONS\n{tier_guidance}"

    manifest = "\n".join(f"- {u['unit_id']}: {u.get('preview', '')[:400]}" for u in units)
    user_content = (
        f"TASK\n{task_description}\n\n"
        f"OUTPUT SCHEMA EACH WORKER MUST SATISFY\n{output_schema}\n\n"
        f"UNIT MANIFEST ({len(units)} units)\n{manifest}"
    )

    result = await call_json(
        model=model,
        system=system,
        user_content=user_content,
        schema=_plan_schema(allow_openrouter),
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
