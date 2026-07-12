"""The scorecard (docs/design.html §5): a live, queryable run ledger.

SQLite-backed so it's queryable mid-run from a second process (`ringer
report <db> <run_id>`) rather than only as a post-hoc summary object.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

from .models import OPENROUTER_MODELS
from .spec import Attempt, UnitStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task_description TEXT,
    started_at REAL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS units (
    run_id TEXT,
    unit_id TEXT,
    status TEXT,
    PRIMARY KEY (run_id, unit_id)
);
CREATE TABLE IF NOT EXISTS attempts (
    run_id TEXT,
    unit_id TEXT,
    attempt_number INTEGER,
    stage TEXT,          -- 'intake' | 'planner' | 'worker' | 'checker' | 'judge'
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost REAL,
    passed INTEGER,
    reason TEXT,
    ts REAL
);
"""


@dataclass
class CostSummary:
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    by_stage: dict[str, float]


@dataclass
class ModelUsage:
    stage: str
    model: str
    calls: int
    cost: float
    input_tokens: int
    output_tokens: int


class Scorecard:
    def __init__(self, db_path: str = "ringer_scorecard.sqlite3"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def start_run(self, task_description: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO runs (run_id, task_description, started_at, finished_at) VALUES (?, ?, ?, NULL)",
            (run_id, task_description, time.time()),
        )
        self._conn.commit()
        return run_id

    def finish_run(self, run_id: str) -> None:
        self._conn.execute("UPDATE runs SET finished_at = ? WHERE run_id = ?", (time.time(), run_id))
        self._conn.commit()

    def record_intake_cost(self, run_id: str, *, model: str, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Record the upfront `ringer intake` proposal call against this run.

        Uses a sentinel unit_id ('__intake__') rather than a new table --
        the intake call isn't tied to any one unit, but it's still a real
        call worth including in the run's total_cost so `ringer report`
        reflects the whole prompt -> execution episode, not just execution.
        Never appears in status_counts()/pass-rate since it never touches
        the `units` table.
        """
        self._conn.execute(
            "INSERT INTO attempts VALUES (?, '__intake__', 0, 'intake', ?, ?, ?, ?, NULL, NULL, ?)",
            (run_id, model, input_tokens, output_tokens, cost, time.time()),
        )
        self._conn.commit()

    def record_planner_cost(self, run_id: str, *, model: str, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Record the once-per-run planning call. Same sentinel-unit_id
        pattern as record_intake_cost -- this call was previously computed
        (planner.PlanResult.cost) but never persisted anywhere, so every
        run's reported total silently excluded it.
        """
        self._conn.execute(
            "INSERT INTO attempts VALUES (?, '__planner__', 0, 'planner', ?, ?, ?, ?, NULL, NULL, ?)",
            (run_id, model, input_tokens, output_tokens, cost, time.time()),
        )
        self._conn.commit()

    def record_worker_attempt(self, run_id: str, attempt: Attempt) -> None:
        self._conn.execute(
            "INSERT INTO attempts VALUES (?, ?, ?, 'worker', ?, ?, ?, ?, NULL, NULL, ?)",
            (
                run_id,
                attempt.unit_id,
                attempt.attempt_number,
                attempt.model,
                attempt.input_tokens,
                attempt.output_tokens,
                attempt.cost,
                time.time(),
            ),
        )
        if attempt.checker_passed is not None:
            self._conn.execute(
                "INSERT INTO attempts VALUES (?, ?, ?, 'checker', NULL, 0, 0, 0, ?, ?, ?)",
                (
                    run_id,
                    attempt.unit_id,
                    attempt.attempt_number,
                    int(attempt.checker_passed),
                    attempt.checker_reason,
                    time.time(),
                ),
            )
        if attempt.judge_passed is not None:
            self._conn.execute(
                "INSERT INTO attempts VALUES (?, ?, ?, 'judge', ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    attempt.unit_id,
                    attempt.attempt_number,
                    attempt.judge_model,
                    attempt.judge_input_tokens,
                    attempt.judge_output_tokens,
                    attempt.judge_cost,
                    int(attempt.judge_passed),
                    attempt.judge_reason,
                    time.time(),
                ),
            )
        self._conn.commit()

    def record_unit_status(self, run_id: str, unit_id: str, status: UnitStatus) -> None:
        self._conn.execute(
            "INSERT INTO units (run_id, unit_id, status) VALUES (?, ?, ?) "
            "ON CONFLICT(run_id, unit_id) DO UPDATE SET status = excluded.status",
            (run_id, unit_id, status.value),
        )
        self._conn.commit()

    def cost_summary(self, run_id: str) -> CostSummary:
        cur = self._conn.execute(
            "SELECT stage, SUM(cost), SUM(input_tokens), SUM(output_tokens) FROM attempts "
            "WHERE run_id = ? GROUP BY stage",
            (run_id,),
        )
        rows = cur.fetchall()
        by_stage = {stage: cost or 0.0 for stage, cost, _, _ in rows}
        total_cost = sum(by_stage.values())
        total_in = sum(r[2] or 0 for r in rows)
        total_out = sum(r[3] or 0 for r in rows)
        return CostSummary(total_cost, total_in, total_out, by_stage)

    def status_counts(self, run_id: str) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT status, COUNT(*) FROM units WHERE run_id = ? GROUP BY status", (run_id,)
        )
        return dict(cur.fetchall())

    def model_summary(self, run_id: str) -> list[ModelUsage]:
        """Which model actually served each stage, and how much it cost.

        Excludes 'checker' rows (model is always NULL there -- it's
        deterministic, no model call) and any worker-error placeholder rows
        (model is '' when a worker call raised before a model could serve
        it at all -- see orchestrator._run_unit's exception handler).
        """
        cur = self._conn.execute(
            "SELECT stage, model, COUNT(*), SUM(cost), SUM(input_tokens), SUM(output_tokens) "
            "FROM attempts WHERE run_id = ? AND model IS NOT NULL AND model != '' "
            "GROUP BY stage, model ORDER BY stage, model",
            (run_id,),
        )
        return [
            ModelUsage(stage, model, count, cost or 0.0, in_tok or 0, out_tok or 0)
            for stage, model, count, cost, in_tok, out_tok in cur.fetchall()
        ]

    def openrouter_models_used(self, run_id: str) -> list[str]:
        """Which OpenRouter models actually served a worker call this run --
        distinct from OrchestratorConfig.allow_openrouter or run_report
        .openrouter_available, both of which describe whether OpenRouter was
        *reachable*, not whether anything actually ended up routed there.
        """
        return sorted(
            {u.model for u in self.model_summary(run_id) if u.stage == "worker" and u.model in OPENROUTER_MODELS}
        )

    def print_report(self, run_id: str) -> None:
        counts = self.status_counts(run_id)
        summary = self.cost_summary(run_id)
        models = self.model_summary(run_id)
        openrouter_used = self.openrouter_models_used(run_id)
        total_units = sum(counts.values()) or 1
        passed = counts.get(UnitStatus.PASSED.value, 0)

        print(f"\n=== ringer scorecard :: run {run_id} ===")
        print(f"units:      {total_units}")
        print(f"pass rate:  {passed}/{total_units} ({passed / total_units:.0%})")
        for status, count in sorted(counts.items()):
            print(f"  {status:<22} {count}")
        print(f"\ntotal cost: ${summary.total_cost:.4f}")
        for stage, cost in sorted(summary.by_stage.items()):
            print(f"  {stage:<10} ${cost:.4f}")
        print(f"tokens:     {summary.total_input_tokens:,} in / {summary.total_output_tokens:,} out")
        if models:
            print("\nmodels:")
            for m in models:
                unit = "call" if m.calls == 1 else "calls"
                print(f"  {m.stage:<10} {m.model:<28} {m.calls} {unit:<5} ${m.cost:.4f}")
        print(f"\nopenrouter: {'yes -- ' + ', '.join(openrouter_used) if openrouter_used else 'not used'}")
        print()

    def close(self) -> None:
        self._conn.close()


@contextmanager
def open_scorecard(db_path: str = "ringer_scorecard.sqlite3"):
    sc = Scorecard(db_path)
    try:
        yield sc
    finally:
        sc.close()
