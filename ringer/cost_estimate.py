"""A rough pre-execution cost estimate -- shown before you confirm, not a forecast guarantee.

Everything the scorecard (scorecard.py) reports is *actual* cost, computed
from real token usage after a call already happened. This module instead
estimates cost *before* anything runs, from information that's cheap to
get: the character length of each unit's raw content, converted to a token
count via a fixed chars-per-token heuristic (no model call, no
`count_tokens` API round trip). It is deliberately conservative and
deliberately labeled as an estimate everywhere it's surfaced -- retries,
judge escalation, and the model's actual verbosity all move the real
number, sometimes substantially.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Role, Tier, estimate_cost, resolve_model

# A rough, widely-used approximation -- see shared/token-counting.md in the
# claude-api skill for why this is fine for a heads-up estimate and not fine
# for anything billing-accurate (use messages.count_tokens for that).
_CHARS_PER_TOKEN = 4

# Fixed system-prompt/instructions/schema overhead per worker call, on top
# of the unit's own content -- worker.py's system prompt plus the rendered
# output_schema plus any failure-context boilerplate on a retry.
_WORKER_OVERHEAD_TOKENS = 400

# Structured-extraction outputs are typically much smaller than their
# source text; this ratio is a floor-adjusted fraction of estimated input.
_OUTPUT_TOKENS_FLOOR = 150
_OUTPUT_TOKENS_RATIO = 0.25

# Planner: reads a short preview per unit (previews are truncated to ~400
# chars elsewhere in the codebase) and writes one spec per unit.
_PLANNER_PREVIEW_TOKENS_CAP = 100
_PLANNER_BASE_OVERHEAD_TOKENS = 300
_PLANNER_OUTPUT_TOKENS_PER_UNIT = 150
_PLANNER_OUTPUT_BASE_TOKENS = 200

# Judge: a much smaller call than a worker's -- spec + one unit's output,
# no source material to re-read.
_JUDGE_INPUT_TOKENS_PER_UNIT = 400
_JUDGE_OUTPUT_TOKENS_PER_UNIT = 150


def _estimate_tokens(content: Any) -> int:
    return max(1, len(str(content)) // _CHARS_PER_TOKEN)


@dataclass
class ExecutionCostEstimate:
    unit_count: int
    planner_model: str
    worker_model: str
    judge_model: str
    planner_cost: float
    worker_cost_single_pass: float
    worker_cost_worst_case: float
    judge_cost_if_all_escalate: float
    low_estimate: float
    high_estimate: float


def estimate_execution_cost(
    units: dict[str, Any],
    *,
    planner_tier: Tier = Tier.DEFAULT,
    worker_tier: Tier = Tier.DEFAULT,
    judge_tier: Tier = Tier.DEFAULT,
    max_retries: int = 3,
) -> ExecutionCostEstimate:
    """Rough cost estimate for running `units` through the harness.

    `low_estimate` assumes every unit passes its checker on the first
    attempt and none need judge review. `high_estimate` assumes every unit
    burns its full retry budget *and* every unit escalates to the judge --
    a genuine worst case, not a typical one. Actual cost usually lands
    much closer to `low_estimate` than `high_estimate` for a well-checked
    task; a wide gap between the two is itself informative (it means
    retries and judge calls, if they happen, are expensive relative to a
    single pass).
    """
    planner_model = resolve_model(Role.PLANNER, planner_tier)
    worker_model = resolve_model(Role.WORKER, worker_tier)
    judge_model = resolve_model(Role.JUDGE, judge_tier)

    unit_count = len(units)
    if unit_count == 0:
        return ExecutionCostEstimate(0, planner_model, worker_model, judge_model, 0, 0, 0, 0, 0, 0)

    planner_input = sum(min(_estimate_tokens(v), _PLANNER_PREVIEW_TOKENS_CAP) for v in units.values())
    planner_input += _PLANNER_BASE_OVERHEAD_TOKENS
    planner_output = _PLANNER_OUTPUT_TOKENS_PER_UNIT * unit_count + _PLANNER_OUTPUT_BASE_TOKENS
    planner_cost = estimate_cost(planner_model, planner_input, planner_output)

    worker_input_total = sum(_estimate_tokens(v) + _WORKER_OVERHEAD_TOKENS for v in units.values())
    worker_output_total = sum(
        max(_OUTPUT_TOKENS_FLOOR, int(_estimate_tokens(v) * _OUTPUT_TOKENS_RATIO)) for v in units.values()
    )
    worker_cost_single_pass = estimate_cost(worker_model, worker_input_total, worker_output_total)
    worker_cost_worst_case = worker_cost_single_pass * (max_retries + 1)

    judge_cost_if_all_escalate = estimate_cost(
        judge_model, _JUDGE_INPUT_TOKENS_PER_UNIT * unit_count, _JUDGE_OUTPUT_TOKENS_PER_UNIT * unit_count
    )

    low_estimate = planner_cost + worker_cost_single_pass
    high_estimate = planner_cost + worker_cost_worst_case + judge_cost_if_all_escalate

    return ExecutionCostEstimate(
        unit_count=unit_count,
        planner_model=planner_model,
        worker_model=worker_model,
        judge_model=judge_model,
        planner_cost=planner_cost,
        worker_cost_single_pass=worker_cost_single_pass,
        worker_cost_worst_case=worker_cost_worst_case,
        judge_cost_if_all_escalate=judge_cost_if_all_escalate,
        low_estimate=low_estimate,
        high_estimate=high_estimate,
    )
