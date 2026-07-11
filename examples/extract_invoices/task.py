"""A worked 'pile of documents' example: `ringer run examples.extract_invoices.task:build_task`

Scores well against the agent test (docs/design.html §1): many independent
units (each invoice stands alone), checkable against the source text
(every claimed line item must quote it verbatim), and large enough that a
single agent reading all invoices in one context would waste tokens on
work that doesn't need to share context at all.
"""

from __future__ import annotations

from ringer.checker import CheckResult, compose, schema_checker
from ringer.models import Tier
from ringer.orchestrator import OrchestratorConfig
from ringer.spec import TaskSpec

from .data import INVOICES

TASK_DESCRIPTION = (
    "Extract structured line-item data from each vendor invoice. Every line "
    "item must include source_quote: a verbatim excerpt from the invoice "
    "text that supports it -- never a paraphrase or an invented figure. The "
    "stated total_amount must reconcile with the sum of the line items."
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "invoice_number": {"type": "string"},
        "currency": {"type": "string"},
        "total_amount": {"type": "number"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "amount": {"type": "number"},
                    "source_quote": {
                        "type": "string",
                        "description": "Verbatim excerpt from the invoice that supports this line item.",
                    },
                },
                "required": ["description", "amount", "source_quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["vendor", "invoice_number", "currency", "total_amount", "line_items"],
    "additionalProperties": False,
}


def _total_matches_line_items(spec: TaskSpec, output: dict) -> CheckResult:
    line_items = output.get("line_items", [])
    computed = round(sum(item.get("amount", 0) for item in line_items), 2)
    stated = round(output.get("total_amount", 0), 2)
    if abs(computed - stated) > 0.01:
        return CheckResult(
            passed=False,
            reason=f"line items sum to {computed} but total_amount is {stated}",
        )
    return CheckResult(passed=True, reason="total_amount reconciles with line items")


def _quotes_verifiable(spec: TaskSpec, output: dict) -> CheckResult:
    """Every source_quote must actually appear in the invoice text.

    This is the literal implementation of "sources must be attached and
    must match the task; entries that fail are rejected" from the brief --
    no model call, just a substring check against ground truth.
    """
    normalized_text = " ".join(str(spec.input_data).split()).lower()
    for i, item in enumerate(output.get("line_items", [])):
        quote = " ".join(item.get("source_quote", "").split()).lower()
        if not quote or quote not in normalized_text:
            return CheckResult(
                passed=False,
                reason=f"line_item {i} source_quote {quote!r} not found verbatim in invoice text",
            )
    return CheckResult(passed=True, reason="all source_quotes verified verbatim against invoice text")


def build_checker():
    return compose(
        schema_checker(OUTPUT_SCHEMA),
        _total_matches_line_items,
        _quotes_verifiable,
    )


def build_task() -> OrchestratorConfig:
    unit_previews = [
        {"unit_id": unit_id, "preview": text.strip().splitlines()[0]}
        for unit_id, text in INVOICES.items()
    ]

    return OrchestratorConfig(
        task_description=TASK_DESCRIPTION,
        output_schema=OUTPUT_SCHEMA,
        units=dict(INVOICES),
        unit_previews=unit_previews,
        checker=build_checker(),
        planner_tier=Tier.DEFAULT,
        judge_tier=Tier.DEFAULT,
        max_retries=2,
        max_concurrency=4,
        scorecard_path="extract_invoices_scorecard.sqlite3",
    )
