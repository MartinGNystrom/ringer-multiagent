"""Shared data shapes passed between planner, worker, checker, and judge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class TaskSpec:
    """One independently-checkable unit of work, written by the planner.

    The planner writes one of these per unit and never touches the unit's
    source material itself at scale (docs §4: "Planner ... never touches the
    work"). Everything a worker needs to execute the unit lives here.
    """

    unit_id: str
    instructions: str
    input_data: Any
    output_schema: dict | None = None
    tier: str = "default"  # ringer.models.Tier value, resolved by the worker
    needs_judge: bool = False
    judge_rubric: str | None = None
    metadata: dict = field(default_factory=dict)
    # Appended by the orchestrator on retry -- see docs §5 "Retry semantics".
    failure_context: list[str] = field(default_factory=list)


class UnitStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    # Retries exhausted without a pass. Per-attempt detail (which stage
    # rejected it and why) lives on Attempt.checker_reason / judge_reason,
    # not as a separate terminal status -- see orchestrator.py.
    NEEDS_HUMAN = "needs_human"


@dataclass
class Attempt:
    """One worker call against a spec, plus whatever graded it."""

    unit_id: str
    attempt_number: int
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    output: Any
    checker_passed: bool | None = None
    checker_reason: str | None = None
    judge_passed: bool | None = None
    judge_reason: str | None = None
    judge_model: str | None = None
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_cost: float = 0.0


@dataclass
class UnitResult:
    """Final outcome for one unit after retries are exhausted or it passes."""

    unit_id: str
    status: UnitStatus
    attempts: list[Attempt]
    final_output: Any = None

    @property
    def total_cost(self) -> float:
        return sum(a.cost + a.judge_cost for a in self.attempts)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)
