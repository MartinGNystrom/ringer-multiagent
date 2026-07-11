"""`ringer run <module:callable>` -- run a task defined in Python.

Checkers are code, not JSON, so the CLI takes a dotted path to a zero-arg
callable that returns an `OrchestratorConfig` (see examples/extract_invoices
for a worked one) rather than a generic config file.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys

from .orchestrator import OrchestratorConfig, run
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


def _cmd_run(args: argparse.Namespace) -> None:
    factory = _load_callable(args.target)
    config: OrchestratorConfig = factory()
    report = asyncio.run(run(config))

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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
