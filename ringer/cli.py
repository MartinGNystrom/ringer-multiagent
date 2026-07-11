"""`ringer run <module:callable>` -- run a task defined in Python.
`ringer intake "<prompt>" --units <path>` -- propose a plan from a prompt,
confirm it, then run it, without writing any Python at all.

Checkers you hand-write in `ringer run` are code, not JSON; `ringer intake`
instead builds a checker only from the fixed, deterministic factories in
checker.py -- see intake.py's module docstring for why that boundary exists.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from . import intake as intake_mod
from .orchestrator import AgentTestGateError, OrchestratorConfig, run
from .scorecard import Scorecard


def _load_callable(dotted_path: str):
    if ":" not in dotted_path:
        raise SystemExit(f"expected 'module.path:callable_name', got {dotted_path!r}")
    module_name, attr = dotted_path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise SystemExit(f"{module_name!r} has no attribute {attr!r}") from exc


def _print_gate_refusal(exc: AgentTestGateError) -> None:
    result = exc.result
    print(f"\nagent test recommends: {result.label}")
    print(f"reason: {result.reason}")
    print("not running -- pass force=True on the OrchestratorConfig (or --force on `ringer intake`) to override.")


def _cmd_run(args: argparse.Namespace) -> None:
    factory = _load_callable(args.target)
    config: OrchestratorConfig = factory()
    try:
        report = asyncio.run(run(config))
    except AgentTestGateError as exc:
        _print_gate_refusal(exc)
        sys.exit(2)

    # run() closes its own scorecard connection when done; reopen to report.
    sc = Scorecard(config.scorecard_path)
    sc.print_report(report.run_id)
    sc.close()

    if report.needs_human:
        print(f"needs human review: {', '.join(report.needs_human)}")
        sys.exit(1)


def _cmd_report(args: argparse.Namespace) -> None:
    sc = Scorecard(args.db_path)
    sc.print_report(args.run_id)
    sc.close()


def _load_units(units_path: str) -> dict[str, Any]:
    path = Path(units_path)
    if path.is_dir():
        units: dict[str, Any] = {}
        for p in sorted(path.iterdir()):
            if p.is_file():
                units[p.name] = p.read_text(encoding="utf-8", errors="replace")
        if not units:
            raise SystemExit(f"no files found in {units_path!r}")
        return units
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"{units_path!r} must contain a JSON object of unit_id -> content")
        return data
    raise SystemExit(f"--units must be a directory or a .json file, got {units_path!r}")


def _print_intake_plan(plan: intake_mod.IntakePlan, unit_count: int) -> None:
    result = plan.agent_test_result
    print(f"\n=== ringer intake proposal ===")
    print(f"units:      {unit_count}")
    print(f"agent test: {result.label} -- {result.reason}")
    print(f"rationale:  {plan.rationale}")
    print("checks:")
    for desc in plan.check_descriptions:
        print(f"  - {desc}")
    print("proposed output schema:")
    print(json.dumps(plan.output_schema, indent=2))
    print(f"intake call cost: ${plan.cost:.4f} ({plan.model}, {plan.input_tokens} in / {plan.output_tokens} out)")
    print()


def _cmd_intake(args: argparse.Namespace) -> None:
    units = _load_units(args.units)
    unit_previews = [{"unit_id": uid, "preview": str(content)[:200]} for uid, content in units.items()]

    plan = asyncio.run(intake_mod.propose(args.prompt, units))
    _print_intake_plan(plan, len(units))

    if not plan.agent_test_result.should_build_multi_agent and not args.force:
        print("not proceeding -- pass --force to run anyway despite the recommendation above.")
        sys.exit(2)

    if not args.yes:
        answer = input("Proceed with this plan? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted -- no tokens spent beyond the intake call above.")
            return

    config = OrchestratorConfig(
        task_description=plan.prompt,
        output_schema=plan.output_schema,
        units=units,
        unit_previews=unit_previews,
        checker=plan.checker,
        allow_openrouter=args.allow_openrouter,
        agent_test=plan.agent_test_inputs,
        force=args.force,
        scorecard_path=args.scorecard,
    )
    try:
        report = asyncio.run(run(config))
    except AgentTestGateError as exc:
        _print_gate_refusal(exc)
        sys.exit(2)

    sc = Scorecard(config.scorecard_path)
    sc.print_report(report.run_id)
    sc.close()

    if report.needs_human:
        print(f"needs human review: {', '.join(report.needs_human)}")
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ringer")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run a task defined by a Python factory function")
    run_parser.add_argument("target", help="dotted path, e.g. examples.extract_invoices.task:build_task")
    run_parser.set_defaults(func=_cmd_run)

    report_parser = sub.add_parser("report", help="print the scorecard for a past run")
    report_parser.add_argument("db_path")
    report_parser.add_argument("run_id")
    report_parser.set_defaults(func=_cmd_report)

    intake_parser = sub.add_parser(
        "intake", help="propose a plan from a prompt, confirm it, then run it -- no Python required"
    )
    intake_parser.add_argument("prompt", help="what you want done, in plain English")
    intake_parser.add_argument(
        "--units", required=True, help="a directory of files, or a .json file of {unit_id: content}"
    )
    intake_parser.add_argument(
        "--allow-openrouter", action="store_true", help="let the planner route some units to OpenRouter open-weight workers"
    )
    intake_parser.add_argument(
        "--force", action="store_true", help="run even if the agent test doesn't recommend multi-agent"
    )
    intake_parser.add_argument("--yes", action="store_true", help="skip the interactive y/n confirmation")
    intake_parser.add_argument("--scorecard", default="ringer_intake_scorecard.sqlite3")
    intake_parser.set_defaults(func=_cmd_intake)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
