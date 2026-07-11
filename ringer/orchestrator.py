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
from typing import Any, Callable

EventCallback = Callable[[str], None]

from . import _openrouter_client
from . import judge as judge_mod
from . import planner as planner_mod
from . import worker as worker_mod
from .agent_test import AgentTestInputs, AgentTestResult
from .agent_test import score as score_agent_test
from .checker import Checker
from .models import Tier
from .scorecard import CostSummary, Scorecard
from .spec import Attempt, TaskSpec, UnitResult, UnitStatus


@dataclass
class IntakeCostRecord:
    """The cost of an upstream `ringer intake` proposal call, passed through
    so run() can persist it into the scorecard alongside execution costs --
    see intake.py / cli.py `ringer intake`. None if a task was hand-written
    via `ringer run` with no intake step.
    """

    model: str
    input_tokens: int
    output_tokens: int
    cost: float


class AgentTestGateError(RuntimeError):
    """Raised when a task's own agent-test score doesn't clear the bar for
    multi-agent execution (docs/design.html §1/§7) and OrchestratorConfig
    .force wasn't set. Carries the full AgentTestResult so a caller (e.g.
    the CLI) can show the verdict and reasoning instead of a bare crash --
    the whole point is to stop *before* any planner or worker tokens are
    spent, not just log a complaint after the fact.
    """

    def __init__(self, result: AgentTestResult):
        self.result = result
        super().__init__(
            f"agent test recommends {result.label!r}, not multi-agent: {result.reason} "
            f"(pass force=True on OrchestratorConfig to run anyway)"
        )


@dataclass
class RunReport:
    run_id: str
    units: list[UnitResult]
    # Authoritative cost across every stage of this run -- intake (if any),
    # planner, workers, and judge. Pulled from the scorecard itself rather
    # than summed from `units`, which only ever covered worker+judge cost
    # and silently excluded the once-per-run planner call.
    cost_summary: CostSummary
    # Whether OpenRouter was actually available to the planner this run --
    # i.e. config.allow_openrouter was True *and* OPENROUTER_API_KEY was
    # set. False whenever the task didn't ask for it, and also False when
    # it asked but the key wasn't configured -- callers that want to tell
    # those two apart should check config.allow_openrouter themselves.
    openrouter_available: bool = False

    @property
    def total_cost(self) -> float:
        return self.cost_summary.total_cost

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
    # Requests that the planner may route units to OpenRouter open-weight
    # workers as a cost tier -- but this is necessary, not sufficient. run()
    # only honors it if OPENROUTER_API_KEY is actually set in the
    # environment (_openrouter_client.is_configured()); otherwise it's
    # silently treated as False and the planner never even sees the
    # OpenRouter tiers, regardless of what this task requests. An
    # environment that never provisions the key is Anthropic-only by
    # construction, not by every task remembering not to opt in.
    allow_openrouter: bool = False
    max_retries: int = 3
    max_concurrency: int = 8
    worker_effort: str = "medium"
    scorecard_path: str = "ringer_scorecard.sqlite3"
    # When set, run() scores this against the agent test (§1) *before*
    # spending a single token on planning, and refuses to proceed unless the
    # verdict clears multi-agent -- or force=True overrides the refusal.
    # None (the default) skips the gate entirely, matching prior behavior.
    agent_test: AgentTestInputs | None = None
    force: bool = False
    # Set by `ringer intake` when this config was built from a prompt
    # proposal, so its cost is persisted alongside execution -- see
    # IntakeCostRecord above. None for a hand-written OrchestratorConfig.
    intake_cost: IntakeCostRecord | None = None


async def _run_unit(
    spec: TaskSpec,
    checker: Checker,
    judge_tier: Tier,
    max_retries: int,
    run_id: str,
    scorecard: Scorecard,
    semaphore: asyncio.Semaphore,
    emit: EventCallback,
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
                emit(f"[{spec.unit_id}] attempt {attempt_number}: worker error -- {exc}")
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
                emit(f"[{spec.unit_id}] attempt {attempt_number}: checker rejected -- {check.reason}")
                continue

            if check.needs_judge or spec.needs_judge:
                emit(f"[{spec.unit_id}] attempt {attempt_number}: checker passed, escalating to judge")
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
                    emit(f"[{spec.unit_id}] judge approved -- PASSED ({attempt_number} attempt(s))")
                    return UnitResult(spec.unit_id, UnitStatus.PASSED, attempts, work.output)

                spec.failure_context.append(f"judge: {verdict.reason}")
                emit(f"[{spec.unit_id}] attempt {attempt_number}: judge rejected -- {verdict.reason}")
                continue

            attempts.append(attempt)
            scorecard.record_worker_attempt(run_id, attempt)
            scorecard.record_unit_status(run_id, spec.unit_id, UnitStatus.PASSED)
            emit(f"[{spec.unit_id}] checker passed -- PASSED ({attempt_number} attempt(s))")
            return UnitResult(spec.unit_id, UnitStatus.PASSED, attempts, work.output)

    # Retries exhausted without a pass -- surfaced for a human, never silently
    # dropped or silently accepted. Per-attempt failure detail (checker vs.
    # judge, and why) is already in `attempts`; the terminal status is the
    # same either way.
    last = attempts[-1] if attempts else None
    scorecard.record_unit_status(run_id, spec.unit_id, UnitStatus.NEEDS_HUMAN)
    emit(f"[{spec.unit_id}] NEEDS HUMAN REVIEW after {len(attempts)} attempt(s) -- {last.checker_reason if last else 'no attempts recorded'}")
    return UnitResult(spec.unit_id, UnitStatus.NEEDS_HUMAN, attempts, last.output if last else None)


async def run(config: OrchestratorConfig, on_event: EventCallback | None = None) -> RunReport:
    """Run a task end to end. `on_event` is an optional callback invoked with
    a human-readable string as things happen (plan complete, each unit's
    attempt result, each unit reaching a terminal state) -- pass it to get
    live progress instead of silence until the whole run finishes. None (the
    default) means no callback is invoked at all, matching prior behavior --
    library callers that don't want console output don't have to opt out of
    anything.
    """
    emit: EventCallback = on_event if on_event is not None else (lambda _msg: None)

    if config.agent_test is not None:
        result = score_agent_test(config.agent_test)
        if not result.should_build_multi_agent and not config.force:
            raise AgentTestGateError(result)

    scorecard = Scorecard(config.scorecard_path)
    run_id = scorecard.start_run(config.task_description)
    emit(f"run {run_id}: {len(config.units)} units, planning...")

    if config.intake_cost is not None:
        scorecard.record_intake_cost(
            run_id,
            model=config.intake_cost.model,
            input_tokens=config.intake_cost.input_tokens,
            output_tokens=config.intake_cost.output_tokens,
            cost=config.intake_cost.cost,
        )

    # Necessary-but-not-sufficient: a task can request OpenRouter, but the
    # environment decides whether it's actually reachable. The planner is
    # never even told OpenRouter tiers exist unless both are true -- this
    # is what makes an environment without OPENROUTER_API_KEY provably
    # Anthropic-only, rather than merely "Anthropic-only as long as no task
    # opts in by mistake."
    effective_allow_openrouter = config.allow_openrouter and _openrouter_client.is_configured()

    plan_result = await planner_mod.plan(
        task_description=config.task_description,
        output_schema=config.output_schema,
        units=config.unit_previews,
        tier=config.planner_tier,
        allow_openrouter=effective_allow_openrouter,
    )
    scorecard.record_planner_cost(
        run_id,
        model=plan_result.model,
        input_tokens=plan_result.input_tokens,
        output_tokens=plan_result.output_tokens,
        cost=plan_result.cost,
    )
    emit(f"planned {len(plan_result.specs)} units via {plan_result.model} (${plan_result.cost:.4f}) -- starting workers")

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
                emit,
            )
            for spec in plan_result.specs
        ]
    )

    scorecard.finish_run(run_id)
    cost_summary = scorecard.cost_summary(run_id)
    scorecard.close()
    passed = sum(1 for u in unit_results if u.status == UnitStatus.PASSED)
    emit(f"run {run_id} complete: {passed}/{len(unit_results)} passed, ${cost_summary.total_cost:.4f} total")
    return RunReport(
        run_id=run_id,
        units=list(unit_results),
        cost_summary=cost_summary,
        openrouter_available=effective_allow_openrouter,
    )
