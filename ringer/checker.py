"""Mechanical checkers (docs/design.html §4/§5).

A Checker is deterministic and cheap -- ideally no model call at all. It
runs on every worker result before the judge is ever considered. A checker
that needs an LLM call to render a verdict belongs in judge.py, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import jsonschema


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str
    # If True, a checker-passed result still needs subjective judge review
    # (docs §5: "only sees checker-passed work needing subjective review").
    needs_judge: bool = False


class Checker(Protocol):
    def __call__(self, spec: "TaskSpec", output: Any) -> CheckResult: ...  # noqa: F821


def schema_checker(schema: dict) -> Checker:
    """Reject outputs that don't validate against a JSON schema."""

    def _check(spec, output: Any) -> CheckResult:  # noqa: ANN001
        try:
            jsonschema.validate(output, schema)
        except jsonschema.ValidationError as exc:
            path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
            return CheckResult(
                passed=False,
                reason=f"schema validation failed at {path}: {exc.message}",
            )
        return CheckResult(passed=True, reason="schema valid")

    return _check


def source_attachment_checker(
    claim_field: str, source_field: str, known_sources: set[str] | None = None
) -> Checker:
    """Reject outputs whose claims don't carry a matching source.

    Every finding/claim in output[claim_field] (a list of dicts) must include
    a non-empty output[claim_field][i][source_field] naming a source. If
    known_sources is given, that source must actually exist in it -- this is
    the harness's literal implementation of "sources must be attached and
    must match the task" from the brief.
    """

    def _check(spec, output: Any) -> CheckResult:  # noqa: ANN001
        claims = output.get(claim_field) if isinstance(output, dict) else None
        if claims is None:
            return CheckResult(passed=False, reason=f"missing '{claim_field}' in output")
        for i, claim in enumerate(claims):
            source = claim.get(source_field) if isinstance(claim, dict) else None
            if not source:
                return CheckResult(
                    passed=False,
                    reason=f"claim {i} has no '{source_field}'",
                )
            if known_sources is not None and source not in known_sources:
                return CheckResult(
                    passed=False,
                    reason=f"claim {i} cites unknown source '{source}'",
                )
        return CheckResult(passed=True, reason="all claims sourced", needs_judge=False)

    return _check


def exit_code_checker(runner: Callable[[Any], int]) -> Checker:
    """Reject outputs whose associated command/test run exits non-zero.

    `runner` takes the worker output and returns a process exit code --
    e.g. running a test suite the worker's patch is supposed to satisfy.
    """

    def _check(spec, output: Any) -> CheckResult:  # noqa: ANN001
        code = runner(output)
        if code != 0:
            return CheckResult(passed=False, reason=f"exit code {code}")
        return CheckResult(passed=True, reason="exit code 0")

    return _check


def compose(*checkers: Checker) -> Checker:
    """Run several checkers in order; fail fast on the first rejection."""

    def _check(spec, output: Any) -> CheckResult:  # noqa: ANN001
        needs_judge = False
        for checker in checkers:
            result = checker(spec, output)
            if not result.passed:
                return result
            needs_judge = needs_judge or result.needs_judge
        return CheckResult(passed=True, reason="all checks passed", needs_judge=needs_judge)

    return _check


@dataclass
class AlwaysNeedsJudge:
    """Marks every checker-passed result as needing subjective judge review.

    Wrap around another checker when the unit type is inherently subjective
    (tone, completeness of an argument) and mechanical checks can only rule
    out the clear failures, never confirm quality.
    """

    inner: Checker

    def __call__(self, spec, output: Any) -> CheckResult:  # noqa: ANN001
        result = self.inner(spec, output)
        if not result.passed:
            return result
        return CheckResult(passed=True, reason=result.reason, needs_judge=True)
