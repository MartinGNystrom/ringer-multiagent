"""A worked 'pile of documents' example: `ringer run examples.extract_invoices.task:build_task`

Scores well against the agent test (docs/design.html §1): many independent
units (each invoice stands alone), checkable against the source text
(every claimed line item must quote it verbatim), and large enough that a
single agent reading all invoices in one context would waste tokens on
work that doesn't need to share context at all.
"""

from __future__ import annotations

from ringer.agent_test import AgentTestInputs
from ringer.checker import compose, numeric_reconciliation_checker, schema_checker, verbatim_quote_checker
from ringer.models import Tier
from ringer.orchestrator import OrchestratorConfig

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


def build_checker():
    """Schema + totals-reconcile + every quote is verbatim in the source --
    the literal implementation of "sources must be attached and must match
    the task; entries that fail are rejected" from the brief. All three are
    generic factories from checker.py; nothing here is invoice-specific
    except the field names.
    """
    return compose(
        schema_checker(OUTPUT_SCHEMA),
        numeric_reconciliation_checker(
            total_field="total_amount", parts_field="line_items", amount_field="amount"
        ),
        verbatim_quote_checker(claims_field="line_items", quote_field="source_quote"),
    )


def build_task(allow_openrouter: bool = False) -> OrchestratorConfig:
    """allow_openrouter=True lets the planner route some invoices to the
    OpenRouter open-weight worker tiers (GLM 5.2 / Kimi K2) instead of
    Sonnet/Haiku -- requires OPENROUTER_API_KEY. The same checker gates the
    output either way, so this is a real head-to-head of the two worker
    backends on identical units, not a separate code path.
    """
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
        allow_openrouter=allow_openrouter,
        max_retries=2,
        max_concurrency=4,
        scorecard_path="extract_invoices_scorecard.sqlite3",
        # docs/design.html §1 scoring for this task shape at production
        # scale (hundreds of invoices, not just these four demo ones):
        # independent per-invoice, strongly checkable (schema + totals +
        # verbatim quotes), low separation-of-concerns need, comes up often
        # enough and matters enough to be worth the token spend.
        agent_test=AgentTestInputs(
            size=3, independence=4, separation=1, checkability=4, frequency=2, value=2
        ),
    )


def build_task_openrouter() -> OrchestratorConfig:
    """Zero-arg variant for the CLI: `ringer run examples.extract_invoices.task:build_task_openrouter`"""
    return build_task(allow_openrouter=True)
