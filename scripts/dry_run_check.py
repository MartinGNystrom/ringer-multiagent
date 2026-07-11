"""Offline sanity check for the orchestrator wiring -- no live API calls.

Monkeypatches planner.plan / worker.run / judge.evaluate with deterministic
fakes so the retry loop, concurrency, and scorecard recording can be
verified without an ANTHROPIC_API_KEY. Not a substitute for a real run
against the API -- see README.md for that.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ringer import orchestrator, planner, worker, judge
from ringer.checker import CheckResult
from ringer.models import MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET
from ringer.orchestrator import OrchestratorConfig
from ringer.spec import TaskSpec, UnitStatus

OUTPUT_SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}

UNITS = {"u1": "one", "u2": "two-fails-once", "u3": "three-always-fails"}
PREVIEWS = [{"unit_id": k, "preview": v} for k, v in UNITS.items()]

attempt_counts: dict[str, int] = {}


async def fake_plan(*, task_description, output_schema, units, tier, **kw):
    specs = [
        TaskSpec(unit_id=u["unit_id"], instructions="double the length", input_data=None, output_schema=OUTPUT_SCHEMA)
        for u in units
    ]
    return planner.PlanResult(specs=specs, input_tokens=100, output_tokens=50, cost=0.001, model=MODEL_OPUS)


async def fake_worker_run(spec, **kw):
    attempt_counts[spec.unit_id] = attempt_counts.get(spec.unit_id, 0) + 1
    n = attempt_counts[spec.unit_id]
    output = {"n": n}
    return worker.WorkResult(output=output, input_tokens=10, output_tokens=5, cost=0.0001, model=MODEL_SONNET)


def fake_checker(spec, output):
    unit_id = spec.unit_id
    n = attempt_counts[unit_id]
    if unit_id == "u1":
        return CheckResult(passed=True, reason="ok first try")
    if unit_id == "u2":
        return CheckResult(passed=(n >= 2), reason="ok on retry" if n >= 2 else "n too low")
    return CheckResult(passed=False, reason="never passes (exercises needs_human path)")


async def fake_judge_evaluate(spec, output, **kw):
    return judge.JudgeResult(passed=True, reason="fine", input_tokens=1, output_tokens=1, cost=0.0, model=MODEL_HAIKU)


async def main():
    planner.plan = fake_plan
    worker.run = fake_worker_run
    judge.evaluate = fake_judge_evaluate
    orchestrator.planner_mod.plan = fake_plan
    orchestrator.worker_mod.run = fake_worker_run
    orchestrator.judge_mod.evaluate = fake_judge_evaluate

    config = OrchestratorConfig(
        task_description="dry run",
        output_schema=OUTPUT_SCHEMA,
        units=UNITS,
        unit_previews=PREVIEWS,
        checker=fake_checker,
        max_retries=2,
        max_concurrency=3,
        scorecard_path="/tmp/ringer_dry_run.sqlite3",
    )

    report = await orchestrator.run(config)

    by_id = {u.unit_id: u for u in report.units}
    assert by_id["u1"].status == UnitStatus.PASSED, by_id["u1"].status
    assert by_id["u1"].attempt_count == 1, by_id["u1"].attempt_count
    assert by_id["u2"].status == UnitStatus.PASSED, by_id["u2"].status
    assert by_id["u2"].attempt_count == 2, by_id["u2"].attempt_count
    assert by_id["u3"].status == UnitStatus.NEEDS_HUMAN, by_id["u3"].status
    assert by_id["u3"].attempt_count == 3, by_id["u3"].attempt_count  # 1 try + 2 retries, cap reached

    print("all orchestrator wiring assertions passed")
    print(f"run_id={report.run_id} pass_rate={report.pass_rate:.0%} needs_human={report.needs_human}")

    from ringer.scorecard import Scorecard

    sc = Scorecard(config.scorecard_path)
    sc.print_report(report.run_id)
    sc.close()


if __name__ == "__main__":
    asyncio.run(main())
