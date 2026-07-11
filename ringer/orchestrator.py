"""Wires planner -> workers -> checker -> judge -> retry -> scorecard.

This is docs/design.html §5 as executable code. Read it alongside the
pipeline diagram there: the planner runs once, workers run concurrently
(bounded by max_concurrency), every result passes the mechanical checker
before the judge is ever considered, and failures loop back with the
specific rejection reason appended -- never a bare retry.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from . import judge as judge_mod
from . import planner as planner_mod
from . import worker as worker_mod
from .checker import Checker
from .models import Tier
from .scorecard import Scorecard
from .spec import Attempt, TaskSpec, UnitResult, UnitStatus


@dataclass
class RunReport:
    run_id: str
    units: list[UnitResult]

    @property
    def total_cost(self) -> float:
        return sum(u.total_cost for u in self.units)

    @property
    def pass_rate(self) -> float:
        if not self.units:
            return 0.0
        passed = sum(1 for u in self.units if u.status == UnitStatus.PASSED)
        return passed / len(self.units)

    @property
    def needs_human(self) -> list[str]:
        return [u.unit_id for u in self.units if u.status == UnitStatus.NEEDS_HUMAN]


@dataclass
class OrchestratorConfig:
    task_description: str
    output_schema: dict
    units: dict[str, Any]  # unit_id -> full input_data, attached after planning
    unit_previews: list[dict]  # [{"unit_id", "preview"}, ...] -- planner sees only this
    checker: Checker
    planner_tier: Tier = Tier.DEFAULT
    judge_tier: Tier = Tier.DEFAULT
    # Let the planner route units to OpenRouter open-weight workers (GLM 5.2,
    # Kimi K2) as a cost tier. Off by default -- see planner.py docstring.
    allow_openrouter: bool = False
    max_retries: int = 3
    max_concurrency: int = 8
    worker_effort: str = "medium"
    scorecard_path: str = "ringer_scorecard.sqlite3"


async def _run_unit(
    spec: TaskSpec,
    checker: Checker,
    judge_tier: Tier,
    max_retries: int,
    run_id: str,
    scorecard: Scorecard,
    semaphore: asyncio.Semaphore,
) -> UnitResult:
    attempts: list[Attempt] = []

    async with semaphore:
        for attempt_number in range(1, max_retries + 2):  # first try + max_retries retries
            try:
                work = await worker_mod.run(spec)
            except Exception as exc:  # noqa: BLE001 -- a worker call can cross into a
                # third-party provider (OpenRouter) with its own failure modes
                # (auth, rate limit, transport error) on top of Anthropic's own
                # (refusal). Any of these is a rejected attempt, not a crashed
                # run -- record it and let the retry loop try again or, once
                # retries are exhausted, surface the unit for a human.
                attempt = Attempt(
                    unit_id=spec.unit_id,
                    attempt_number=attempt_number,
                    model="",
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                    output=None,
                    checker_passed=False,
                    checker_reason=f"worker call raised {type(exc).__name__}: {exc}",
                )
                attempts.append(attempt)
                scorecard.record_worker_attempt(run_id, attempt)
                spec.failure_context.append(f"worker error: {exc}")
                continue

            attempt = Attempt(
                unit_id=spec.unit_id,
                attempt_number=attempt_number,
                model=work.model,
                input_tokens=work.input_tokens,
                output_tokens=work.output_tokens,
                cost=work.cost,
                output=work.output,
            )

            check = checker(spec, work.output)
            attempt.checker_passed = check.passed
            attempt.checker_reason = check.reason

            if not check.passed:
                attempts.append(attempt)
                scorecard.record_worker_attempt(run_id, attempt)
                spec.failure_context.append(f"checker: {check.reason}")
                continue

            if check.needs_judge or spec.needs_judge:
                verdict = await judge_mod.evaluate(spec, work.output, tier=judge_tier)
                attempt.judge_passed = verdict.passed
                attempt.judge_reason = verdict.reason
                attempt.judge_model = verdict.model
                attempt.judge_input_tokens = verdict.input_tokens
                attempt.judge_output_tokens = verdict.output_tokens
                attempt.judge_cost = verdict.cost

                attempts.append(attempt)
                scorecard.record_worker_attempt(run_id, attempt)

                if verdict.passed:
                    scorecard.record_unit_status(run_id, spec.unit_id, UnitStatus.PASSED)
                    return UnitResult(spec.unit_id, UnitStatus.PASSED, attempts, work.output)

                spec.failure_context.append(f"judge: {verdict.reason}")
                continue

            attempts.append(attempt)
            scorecard.record_worker_attempt(run_id, attempt)
            scorecard.record_unit_status(run_id, spec.unit_id, UnitStatus.PASSED)
            return UnitResult(spec.unit_id, UnitStatus.PASSED, attempts, work.output)

    # Retries exhausted without a pass -- surfaced for a human, never silently
    # dropped or silently accepted. Per-attempt failure detail (checker vs.
    # judge, and why) is already in `attempts`; the terminal status is the
    # same either way.
    last = attempts[-1] if attempts else None
    scorecard.record_unit_status(run_id, spec.unit_id, UnitStatus.NEEDS_HUMAN)
    return UnitResult(spec.unit_id, UnitStatus.NEEDS_HUMAN, attempts, last.output if last else None)


async def run(config: OrchestratorConfig) -> RunReport:
    scorecard = Scorecard(config.scorecard_path)
    run_id = scorecard.start_run(config.task_description)

    plan_result = await planner_mod.plan(
        task_description=config.task_description,
        output_schema=config.output_schema,
        units=config.unit_previews,
        tier=config.planner_tier,
        allow_openrouter=config.allow_openrouter,
    )

    for spec in plan_result.specs:
        spec.input_data = config.units.get(spec.unit_id)

    semaphore = asyncio.Semaphore(config.max_concurrency)
    unit_results = await asyncio.gather(
        *[
            _run_unit(
                spec,
                config.checker,
                config.judge_tier,
                config.max_retries,
                run_id,
                scorecard,
                semaphore,
            )
            for spec in plan_result.specs
        ]
    )

    scorecard.finish_run(run_id)
    scorecard.close()
    return RunReport(run_id=run_id, units=list(unit_results))
