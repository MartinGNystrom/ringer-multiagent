# ringer

A router–judge multi-agent harness on the Claude API, built on Nate Jones'
"Ringer" pattern: a strong model plans once and never touches the work, a
swarm of cheap models burns the tokens, a deterministic checker gates every
result before it's trusted, and an expensive judge is rationed for only
what the checker can't decide.

**Read [`docs/design.html`](docs/design.html) first** — it's the design
rationale for every module here, including an interactive calculator for
the "one-minute agent test" that gates whether a task should be multi-agent
at all. This README is setup and usage; the design doc is the *why*.

## Roles → models

| Role | Default | Escalated / thrift |
|---|---|---|
| Planner | `claude-opus-4-8` | `claude-fable-5` for genuinely novel task shapes |
| Worker | `claude-sonnet-5` | `claude-haiku-4-5` for routine, high-volume units |
| Judge | `claude-opus-4-8` | `claude-fable-5` for high-stakes (compliance/legal/financial) units |

See `ringer/models.py` for the resolution table and current pricing.

## Install

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...   # or `ant auth login` — see the claude-api skill
```

## Run the worked example

`examples/extract_invoices` is a small "pile of documents" task: extract
structured line items from four vendor invoices, where every claimed line
item must quote the source invoice verbatim and totals must reconcile —
both checked mechanically, with no model call, before any output is trusted.

```bash
python -m ringer.cli run examples.extract_invoices.task:build_task
```

This runs the full pipeline (plan → workers in parallel → mechanical
checker → judge escalation where flagged → retry with failure context →
scorecard) and prints a cost/pass-rate report. Re-print that report later
with:

```bash
python -m ringer.cli report extract_invoices_scorecard.sqlite3 <run_id>
```

## Build your own task

A task is a zero-argument Python factory returning an
`OrchestratorConfig` — see `examples/extract_invoices/task.py` for a full
worked example. The four things you supply:

1. **`task_description`** — what the pile is and what "done" means per unit.
2. **`output_schema`** — the JSON schema every worker's output must satisfy.
3. **`units` / `unit_previews`** — the full input per unit, and a lightweight
   preview per unit (the planner only ever sees previews — see docs §2 on
   why that's what keeps the planner's own context small regardless of
   pile size).
4. **`checker`** — a deterministic `(spec, output) -> CheckResult` function.
   Compose several with `ringer.checker.compose(...)`; `schema_checker(...)`
   and `source_attachment_checker(...)` are ready-made for the common cases.
   If a unit type is inherently subjective, wrap your checker in
   `AlwaysNeedsJudge(...)` or have the planner flag it — either routes the
   checker-passed result to the judge instead of trusting it outright.

```python
from ringer.checker import compose, schema_checker
from ringer.orchestrator import OrchestratorConfig
from ringer.models import Tier

def build_task() -> OrchestratorConfig:
    return OrchestratorConfig(
        task_description="...",
        output_schema={...},
        units={"unit-1": "<full content>", ...},
        unit_previews=[{"unit_id": "unit-1", "preview": "<short preview>"}, ...],
        checker=compose(schema_checker({...}), my_custom_check),
        planner_tier=Tier.DEFAULT,
        judge_tier=Tier.DEFAULT,
        max_retries=3,
        max_concurrency=8,
        scorecard_path="my_task_scorecard.sqlite3",
    )
```

Then: `python -m ringer.cli run mymodule:build_task`.

## Before building a task at all

Run the agent test in `docs/design.html` §1 (or `ringer.agent_test.score`)
on the task first. It's a gate, not a formality — a task that's small,
entangled, and uncheckable should stay a single chat turn, and the harness
will cost you real tokens decomposing it anyway if you skip the check:

```python
from ringer import AgentTestInputs, score

result = score(AgentTestInputs(size=4, independence=4, separation=1, checkability=3, frequency=2, value=2))
print(result.verdict, "-", result.reason)
```

## Project layout

```
docs/design.html          design spec + interactive agent-test calculator
ringer/
  agent_test.py            the four-question + economics scorer (§1)
  models.py                model IDs, pricing, role → model resolution (§6)
  _client.py                shared Anthropic call plumbing (thinking/effort/fallback rules per model)
  planner.py                plans once, never does the work (§4/§5)
  worker.py                 executes one TaskSpec, self-confidence ignored (§4/§5)
  checker.py                deterministic Checker protocol + example checkers (§4/§5)
  judge.py                  fresh-eyes review, rationed to checker-flagged units (§4/§5)
  orchestrator.py           wires it all together + retry-with-failure-context (§5)
  scorecard.py               SQLite-backed cost/pass-rate ledger (§5)
  cli.py                     `ringer run` / `ringer report`
examples/extract_invoices/  a worked pile-of-documents task, end to end
scripts/dry_run_check.py    offline wiring check (mocks the model calls — no API key needed)
```

## Verifying without spending tokens

```bash
python scripts/dry_run_check.py
```

Monkeypatches the planner/worker/judge calls with deterministic fakes and
asserts the retry loop, concurrency, and scorecard recording all behave —
useful for CI or for checking a change to `orchestrator.py` without paying
for live model calls.
