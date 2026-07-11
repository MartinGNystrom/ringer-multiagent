"""The one-minute agent test (docs/design.html §1), as code.

Four structural 0-4 estimates (size, independence, separation of concerns,
checkability) plus two economic dials (frequency, value) produce a
recommendation: no AI, single agent, single agent deferred pending a
checker, or multi-agent.

This mirrors the JS calculator embedded in the design doc exactly, on
purpose -- the doc's interactive tool and this scorer should never drift
apart. If you change one, change the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    NO_AI = "no_ai"
    SINGLE_AGENT = "single_agent"
    SINGLE_AGENT_NEEDS_CHECKER = "single_agent_needs_checker"
    MULTI_AGENT_ROLES = "multi_agent_roles"
    MULTI_AGENT = "multi_agent"


_VERDICT_LABELS = {
    Verdict.NO_AI: "No AI",
    Verdict.SINGLE_AGENT: "Single agent",
    Verdict.SINGLE_AGENT_NEEDS_CHECKER: "Single agent (for now)",
    Verdict.MULTI_AGENT_ROLES: "Multi-agent (roles)",
    Verdict.MULTI_AGENT: "Multi-agent",
}


@dataclass(frozen=True)
class AgentTestInputs:
    """Four structural estimates (0-4) and two economic dials.

    size: does the task exceed what one agent can hold in context at full
        quality? 0 = a single calendar, 4 = thousands of docs/logs/emails.
    independence: can the pieces be worked without coordinating? 0 = fully
        entangled, 4 = fully independent units.
    separation: does correctness require a mind that didn't produce the
        work (author != critic, entry != approval)? 0 = none needed,
        4 = required, conflict of interest otherwise.
    checkability: is verifying an answer much cheaper than producing one?
        0 = no check available, 4 = cheap, near-total mechanical check.
    frequency: 0 = one-off, 1 = monthly, 2 = weekly, 3 = daily or more.
    value: 0 = low, 1 = moderate, 2 = high, 3 = critical.
    """

    size: int
    independence: int
    separation: int
    checkability: int
    frequency: int = 1
    value: int = 1

    def __post_init__(self) -> None:
        for name in ("size", "independence", "separation", "checkability"):
            v = getattr(self, name)
            if not 0 <= v <= 4:
                raise ValueError(f"{name} must be 0-4, got {v}")
        for name in ("frequency", "value"):
            v = getattr(self, name)
            if not 0 <= v <= 3:
                raise ValueError(f"{name} must be 0-3, got {v}")


@dataclass(frozen=True)
class AgentTestResult:
    verdict: Verdict
    label: str
    reason: str
    inputs: AgentTestInputs

    @property
    def should_build_multi_agent(self) -> bool:
        return self.verdict in (Verdict.MULTI_AGENT, Verdict.MULTI_AGENT_ROLES)


def score(inputs: AgentTestInputs) -> AgentTestResult:
    size, indep, sep, check = (
        inputs.size,
        inputs.independence,
        inputs.separation,
        inputs.checkability,
    )
    budget_ok = (inputs.frequency + inputs.value) >= 2

    if size <= 1 and indep <= 1 and check <= 1 and sep <= 1:
        verdict, reason = (
            Verdict.NO_AI,
            "Small, entangled, and unchecked -- a human just does this faster "
            "than any harness could.",
        )
    elif sep >= 3 and check <= 2:
        verdict, reason = (
            Verdict.MULTI_AGENT_ROLES,
            "The separation-of-concerns signal is high enough that a single "
            "agent reviewing its own work is the actual risk, regardless of "
            "size.",
        )
    elif size <= 1 and indep <= 2:
        verdict, reason = (
            Verdict.SINGLE_AGENT,
            "Fits comfortably in one context and doesn't decompose into "
            "independent chunks -- a team would just add coordination "
            "overhead.",
        )
    elif not budget_ok:
        verdict, reason = (
            Verdict.SINGLE_AGENT,
            "The task shape could support a team, but low frequency and low "
            "value don't justify the extra token spend yet.",
        )
    elif check <= 1:
        verdict, reason = (
            Verdict.SINGLE_AGENT_NEEDS_CHECKER,
            "Large and independent, but with no cheap way to check results "
            "-- spin up workers only once you've built a checker, or you'll "
            "generate a pile you can't grade.",
        )
    else:
        verdict, reason = (
            Verdict.MULTI_AGENT,
            "Large, independent, and checkable -- the classic 'pile' shape "
            "this harness is built for.",
        )

    return AgentTestResult(
        verdict=verdict,
        label=_VERDICT_LABELS[verdict],
        reason=reason,
        inputs=inputs,
    )
