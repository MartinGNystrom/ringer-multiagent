"""Offline sanity check for `ringer intake` -- no live API calls.

Monkeypatches intake's model call and the orchestrator's planner/worker/
judge calls with deterministic fakes, writes a small units directory to a
temp dir, and drives cli._cmd_intake() directly (skipping subprocess/argv
parsing) to verify the whole prompt -> proposal -> confirm -> execute path
wires together. Not a substitute for a real run -- see README.md.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ringer import cli, intake, orchestrator, planner, worker, judge
from ringer.models import MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET
from ringer.spec import TaskSpec, UnitStatus

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"total": {"type": "number"}, "items": {"type": "array"}},
    "required": ["total", "items"],
    "additionalProperties": False,
}


async def fake_intake_call_json(*, model, system, user_content, schema, max_tokens=8000, effort="high"):
    payload = {
        "agent_test": {"size": 3, "independence": 4, "separation": 1, "checkability": 3, "frequency": 2, "value": 2},
        "output_schema_json": json.dumps(OUTPUT_SCHEMA),
        "checks": [],
        "rationale": "Independent receipts, checkable via schema only in this fake.",
    }

    class R:
        parsed = payload
        input_tokens = 300
        output_tokens = 100
        cost = 0.005
        served_by = MODEL_OPUS

    return R()


async def fake_plan(*, task_description, output_schema, units, tier, **kw):
    specs = [
        TaskSpec(unit_id=u["unit_id"], instructions="extract", input_data=None, output_schema=output_schema)
        for u in units
    ]
    return planner.PlanResult(specs=specs, input_tokens=50, output_tokens=20, cost=0.001, model=MODEL_OPUS)


async def fake_worker_run(spec, **kw):
    return worker.WorkResult(
        output={"total": 7, "items": [{"amount": 7}]},
        input_tokens=20,
        output_tokens=10,
        cost=0.0002,
        model=MODEL_SONNET,
    )


async def fake_judge_evaluate(spec, output, **kw):
    return judge.JudgeResult(passed=True, reason="fine", input_tokens=1, output_tokens=1, cost=0.0, model=MODEL_HAIKU)


def main():
    intake.call_json = fake_intake_call_json
    orchestrator.planner_mod.plan = fake_plan
    orchestrator.worker_mod.run = fake_worker_run
    orchestrator.judge_mod.evaluate = fake_judge_evaluate

    with tempfile.TemporaryDirectory() as tmp:
        units_dir = Path(tmp) / "units"
        units_dir.mkdir()
        (units_dir / "receipt1.txt").write_text("Coffee $4, pastry $3, total $7")
        (units_dir / "receipt2.txt").write_text("Lunch $12, tip $2, total $14")

        args = argparse.Namespace(
            prompt="extract itemized totals from these receipts",
            units=str(units_dir),
            allow_openrouter=False,
            force=False,
            yes=True,  # skip interactive input()
            max_retries=3,
            scorecard=str(Path(tmp) / "intake_scorecard.sqlite3"),
            quiet=False,
        )

        try:
            cli._cmd_intake(args)
        except SystemExit as exc:
            assert exc.code in (None, 0), f"unexpected exit code {exc.code}"

    print("INTAKE CLI DRY RUN PASSED")


if __name__ == "__main__":
    main()
