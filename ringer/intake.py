"""Intake: turn a raw prompt into a plan, without ever writing checker code.

This is the front door the design doc doesn't have yet: instead of a
developer hand-writing an OrchestratorConfig (task_description, schema,
checker), you hand `propose()` a prompt and a manifest of raw units, and it
returns an IntakePlan -- the agent-test verdict, a proposed output schema,
and a checker built *only* from a fixed menu of deterministic factories in
checker.py (schema validation, numeric reconciliation, verbatim quotes).

The one thing intake deliberately never does is generate checker code. The
whole harness's trust model rests on the checker being mechanical and
inspectable (docs/design.html §2, §4) -- letting an LLM author the gate
that's supposed to catch the LLM's own mistakes would quietly remove the
one thing standing between "cheap worker" and "cheap worker nobody is
checking." If the menu doesn't cover what a task needs, `checks` comes
back empty and the rationale says so -- schema-only verification, visible
to the human deciding whether to proceed, not a silently invented one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import jsonschema

from ._client import call_json
from .agent_test import AgentTestInputs, AgentTestResult
from .agent_test import score as score_agent_test
from .checker import Checker, compose, numeric_reconciliation_checker, schema_checker, verbatim_quote_checker
from .models import Role, Tier, resolve_model

_INTAKE_SYSTEM = """\
You are the intake stage of a multi-agent harness. Given a task in plain \
English and a manifest of the units of work available (previews only, not \
full content), you decide how this task should be scored and structured. \
You do not do the task yourself and you never write executable code.

Produce:

1. agent_test: your honest estimate of four structural dimensions (0-4 \
   each) and two economic dials (0-3 each):
   - size: 0 = a single small item, 4 = a large pile (hundreds+ of units)
   - independence: 0 = fully entangled, 4 = each unit can be done alone
   - separation: 0 = no conflict-of-interest risk, 4 = the author of a \
     unit must not also be its sole judge of correctness
   - checkability: 0 = no way to verify mechanically, 4 = cheap, thorough \
     mechanical verification is possible against the units' own content
   - frequency: 0 = one-off, 3 = daily or more
   - value: 0 = low stakes if wrong, 3 = critical if wrong

2. output_schema_json: a JSON Schema (draft 2020-12), as a JSON-encoded \
   string (not a nested object), describing exactly what a worker must \
   produce for one unit.

3. checks: zero or more entries picked from this fixed menu only -- never \
   invent a new check type, and leave a field null when it doesn't apply:
   - "numeric_reconciliation": some numeric field must equal the sum of a \
     list of parts (needs total_field, parts_field, amount_field)
   - "verbatim_quote": claims must quote the unit's own input verbatim \
     rather than paraphrase or invent facts (needs claims_field, quote_field)
   If nothing in the menu fits this task, return an empty checks array --
   do not force-fit one. Schema validation always runs regardless.

4. rationale: 2-4 sentences a human can read before deciding whether to \
   proceed -- what shape you think this task has, why, and what (if \
   anything) the mechanical checks won't catch."""

_INTAKE_SCHEMA = {
    "type": "object",
    "properties": {
        "agent_test": {
            "type": "object",
            "properties": {
                "size": {"type": "integer"},
                "independence": {"type": "integer"},
                "separation": {"type": "integer"},
                "checkability": {"type": "integer"},
                "frequency": {"type": "integer"},
                "value": {"type": "integer"},
            },
            "required": ["size", "independence", "separation", "checkability", "frequency", "value"],
            "additionalProperties": False,
        },
        "output_schema_json": {"type": "string"},
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["numeric_reconciliation", "verbatim_quote"]},
                    "total_field": {"type": ["string", "null"]},
                    "parts_field": {"type": ["string", "null"]},
                    "amount_field": {"type": ["string", "null"]},
                    "claims_field": {"type": ["string", "null"]},
                    "quote_field": {"type": ["string", "null"]},
                },
                "required": [
                    "type", "total_field", "parts_field", "amount_field", "claims_field", "quote_field",
                ],
                "additionalProperties": False,
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["agent_test", "output_schema_json", "checks", "rationale"],
    "additionalProperties": False,
}


class IntakeSchemaError(ValueError):
    """The model's proposed output_schema_json wasn't valid JSON or wasn't a valid JSON Schema."""


def _clamp(value: Any, lo: int, hi: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, value))


def _build_checker(output_schema: dict, check_specs: list[dict]) -> tuple[Checker, list[str]]:
    checkers: list[Checker] = [schema_checker(output_schema)]
    descriptions = ["schema validation against the proposed output shape"]
    for spec in check_specs:
        kind = spec.get("type")
        if kind == "numeric_reconciliation" and spec.get("total_field") and spec.get("parts_field") and spec.get("amount_field"):
            checkers.append(
                numeric_reconciliation_checker(
                    total_field=spec["total_field"],
                    parts_field=spec["parts_field"],
                    amount_field=spec["amount_field"],
                )
            )
            descriptions.append(
                f"numeric reconciliation: {spec['parts_field']}[].{spec['amount_field']} must sum to {spec['total_field']}"
            )
        elif kind == "verbatim_quote" and spec.get("claims_field") and spec.get("quote_field"):
            checkers.append(
                verbatim_quote_checker(claims_field=spec["claims_field"], quote_field=spec["quote_field"])
            )
            descriptions.append(
                f"verbatim quote check: {spec['claims_field']}[].{spec['quote_field']} must appear in the unit's own input"
            )
        # Anything else -- an unrecognized type, or one missing its required
        # fields -- is silently skipped. Schema validation still applies;
        # nothing here ever executes model-authored logic.
    return compose(*checkers), descriptions


@dataclass
class IntakePlan:
    prompt: str
    agent_test_inputs: AgentTestInputs
    agent_test_result: AgentTestResult
    output_schema: dict
    checker: Checker
    check_descriptions: list[str]
    rationale: str
    input_tokens: int
    output_tokens: int
    cost: float
    model: str


async def propose(
    prompt: str,
    units: dict[str, Any],
    *,
    tier: Tier = Tier.DEFAULT,
    max_tokens: int = 8000,
    effort: str = "high",
) -> IntakePlan:
    """Turn a prompt + raw units into an IntakePlan -- agent-test verdict,
    proposed schema, and a checker built only from checker.py's fixed
    factories. Does not execute anything; see orchestrator.py for that,
    once a human has reviewed this plan (cli.py `ringer intake` does both).
    """
    model = resolve_model(Role.PLANNER, tier)
    manifest = "\n".join(f"- {unit_id}: {str(content)[:400]}" for unit_id, content in units.items())
    user_content = f"TASK (from the user, verbatim)\n{prompt}\n\nUNIT MANIFEST ({len(units)} units)\n{manifest}"

    result = await call_json(
        model=model,
        system=_INTAKE_SYSTEM,
        user_content=user_content,
        schema=_INTAKE_SCHEMA,
        max_tokens=max_tokens,
        effort=effort,
    )
    raw = result.parsed

    at_raw = raw["agent_test"]
    agent_test_inputs = AgentTestInputs(
        size=_clamp(at_raw["size"], 0, 4),
        independence=_clamp(at_raw["independence"], 0, 4),
        separation=_clamp(at_raw["separation"], 0, 4),
        checkability=_clamp(at_raw["checkability"], 0, 4),
        frequency=_clamp(at_raw["frequency"], 0, 3),
        value=_clamp(at_raw["value"], 0, 3),
    )
    agent_test_result = score_agent_test(agent_test_inputs)

    try:
        output_schema = json.loads(raw["output_schema_json"])
        jsonschema.Draft202012Validator.check_schema(output_schema)
    except (json.JSONDecodeError, jsonschema.SchemaError) as exc:
        raise IntakeSchemaError(f"model produced an invalid JSON schema: {exc}") from exc

    checker, descriptions = _build_checker(output_schema, raw.get("checks", []))

    return IntakePlan(
        prompt=prompt,
        agent_test_inputs=agent_test_inputs,
        agent_test_result=agent_test_result,
        output_schema=output_schema,
        checker=checker,
        check_descriptions=descriptions,
        rationale=raw["rationale"],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost=result.cost,
        model=result.served_by,
    )
