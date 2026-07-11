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

Planner and judge always stay on Anthropic — that part of the Ringer thesis
doesn't change: a strong, trusted model plans and grades. Workers can route
to Anthropic *or*, for the cheapest high-volume units, open-weight models
served through [OpenRouter](https://openrouter.ai).

| Role | Default | Escalated / cheaper tiers |
|---|---|---|
| Planner | `claude-opus-4-8` | `claude-fable-5` for genuinely novel task shapes |
| Worker | `claude-sonnet-5` | `claude-haiku-4-5` (thrift) · via OpenRouter (cheapest, opt-in): `z-ai/glm-5.2` / `deepseek/deepseek-v4-flash` (general-purpose), `moonshotai/kimi-k2.7-code` / `qwen/qwen3-coder` (coding) |
| Judge | `claude-opus-4-8` | `claude-fable-5` for high-stakes (compliance/legal/financial) units |

See `ringer/models.py` for the resolution table and current pricing.

## Install

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...     # or `ant auth login` — see the claude-api skill
export OPENROUTER_API_KEY=sk-or-...     # only needed if you enable the OpenRouter worker tiers
```

### Using the OpenRouter worker tiers (GLM 5.2, DeepSeek V4 Flash, Kimi K2, Qwen3 Coder)

Off by default — a task has to opt in with `allow_openrouter=True` on its
`OrchestratorConfig` (or by having its planner call pass `allow_openrouter=True`
directly), because open-weight models are meaningfully less consistent at
structured output than Anthropic's `output_config.format`. That's a real
trade, not a bug: `ringer/_openrouter_client.py` never raises on a
malformed reply — it hands the raw text to the mechanical checker, which
rejects it like any other bad output, and the harness's own retry loop
(with the parse error appended as failure context, docs §5) gets another
attempt. In practice this means the cheapest tier costs you a few extra
retries on the units it can't produce clean structured output for, not a
crashed run — and the checker is exactly what makes that an acceptable
trade rather than a silent quality regression.

When `allow_openrouter=True`, the *planner* decides per unit whether it's
routine and low-stakes enough to route off Sonnet/Haiku, and which
OpenRouter tier fits: `z-ai/glm-5.2` / `deepseek/deepseek-v4-flash` for
general-purpose prose/extraction units, `moonshotai/kimi-k2.7-code` /
`qwen/qwen3-coder` for coding-shaped units. It's told explicitly to avoid
all four for anything also flagged `needs_judge`, since a judge escalation
implies the checker alone can't confirm quality and the worker's
reliability matters more there. See `examples/extract_invoices/task.py`
→ `build_task_openrouter()` for a runnable head-to-head: same task, same
checker, only the worker backend differs.

Planner/judge deliberately never route to OpenRouter, even though it's
cheap — both are low-volume, high-consequence calls (one bad plan cascades
to every unit; a judge grading its own provider's blind spots isn't fresh
eyes), so the savings would be small while the downside of a less-proven
model in that specific role wouldn't be. Workers are the high-volume side
of the harness, protected by the checker + retry loop, which is why that's
where the cost tier actually lives.

## Run the worked example

`examples/extract_invoices` is a small "pile of documents" task: extract
structured line items from four vendor invoices, where every claimed line
item must quote the source invoice verbatim and totals must reconcile —
both checked mechanically, with no model call, before any output is trusted.

```bash
python -m ringer.cli run examples.extract_invoices.task:build_task

# same task, but the planner may route some invoices to OpenRouter
# open-weight workers (GLM 5.2, DeepSeek V4 Flash, Kimi K2, Qwen3 Coder)
# instead of Sonnet/Haiku:
python -m ringer.cli run examples.extract_invoices.task:build_task_openrouter
```

This runs the full pipeline (plan → workers in parallel → mechanical
checker → judge escalation where flagged → retry with failure context →
scorecard) and prints a cost/pass-rate report. Re-print that report later
with:

```bash
python -m ringer.cli report extract_invoices_scorecard.sqlite3 <run_id>
```

## Prompt-driven flow — no Python required

If you don't want to hand-write an `OrchestratorConfig`, give it a prompt
and a folder of raw material instead:

```bash
ringer intake "extract itemized totals from these receipts" --units ./receipts/
```

This runs a separate intake call (Opus by default) that estimates the
agent-test dimensions from your prompt and a preview of each unit,
proposes an output schema, and picks whatever checks from a fixed menu fit
(currently: schema validation, numeric reconciliation, verbatim-quote
verification). It then prints all of that — verdict, schema, checks, and
the intake call's own cost — and asks you to confirm before spending
anything on actual execution:

```
=== ringer intake proposal ===
units:      12
agent test: Multi-agent -- Large, independent, and checkable...
rationale:  Twelve independent receipts; totals and line items are both
            mechanically verifiable against each receipt's own text.
checks:
  - schema validation against the proposed output shape
  - numeric reconciliation: items[].amount must sum to total
  - verbatim quote check: items[].quote must appear in the unit's own input
proposed output schema: {...}
intake call cost: $0.0091 (claude-opus-4-8, 1840 in / 210 out)

Proceed with this plan? [y/N]:
```

Answering `y` runs the exact same planner → workers → checker → judge →
scorecard pipeline as `ringer run` — `ringer intake` only replaces how the
`OrchestratorConfig` gets built, not what runs it. `--units` takes either a
directory (each file becomes one unit) or a `.json` file of
`{"unit_id": "content", ...}`. `--allow-openrouter` and `--force` mirror
the flags of the same name elsewhere; `--yes` skips the interactive
confirmation for scripting.

**What intake will never do: write your checker for you.** The checker is
the one thing in this design that's supposed to be mechanical and
inspectable — an LLM inventing the gate meant to catch its own mistakes
would quietly undo that. So `ringer/intake.py` only ever composes checks
from `checker.py`'s fixed factories; if your task needs verification the
menu doesn't cover, the proposal says so in its rationale and falls back
to schema-only checking, visible to you before you confirm anything —
never silently invented on your behalf. If your checker needs are more
involved than that menu, write an `OrchestratorConfig` by hand (below);
`ringer intake` isn't trying to replace that path, just avoid it when a
generic check is genuinely all a task needs.

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
from ringer.agent_test import AgentTestInputs
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
        # optional -- see "Before building a task at all" below
        agent_test=AgentTestInputs(size=4, independence=4, separation=1, checkability=3, frequency=2, value=2),
    )
```

Then: `python -m ringer.cli run mymodule:build_task`.

## Before building a task at all

Run the agent test in `docs/design.html` §1 (or `ringer.agent_test.score`)
on the task first. It's a gate, not a formality — a task that's small,
entangled, and uncheckable should stay a single chat turn, and the harness
will cost you real tokens decomposing it anyway if you skip the check.

Score it standalone:

```python
from ringer import AgentTestInputs, score

result = score(AgentTestInputs(size=4, independence=4, separation=1, checkability=3, frequency=2, value=2))
print(result.verdict, "-", result.reason)
```

Or set `agent_test=` on the `OrchestratorConfig` (as above) and it becomes
an *enforced* gate: `orchestrator.run()` scores it before the planner is
ever invoked and raises `AgentTestGateError` if the verdict doesn't clear
multi-agent, unless you also pass `force=True`. `ringer intake` (above)
does this automatically for every prompt-driven run — you never have to
remember to check it yourself.

## Project layout

```
docs/design.html          design spec + interactive agent-test calculator
ringer/
  agent_test.py            the four-question + economics scorer (§1)
  models.py                model IDs, pricing, providers, role → model resolution (§6)
  _client.py                shared Anthropic call plumbing (thinking/effort/fallback rules per model)
  _openrouter_client.py      OpenRouter call plumbing for open-weight workers (GLM 5.2, DeepSeek V4 Flash, Kimi K2, Qwen3 Coder)
  planner.py                plans once, never does the work (§4/§5)
  worker.py                 executes one TaskSpec against either backend, self-confidence ignored (§4/§5)
  checker.py                deterministic Checker protocol + example checkers (§4/§5)
  judge.py                  fresh-eyes review, rationed to checker-flagged units (§4/§5)
  orchestrator.py           wires it all together + retry-with-failure-context + the agent-test gate (§5)
  intake.py                 prompt + raw units → agent-test estimate, proposed schema, checker (menu only)
  scorecard.py               SQLite-backed cost/pass-rate ledger (§5)
  cli.py                     `ringer run` / `ringer intake` / `ringer report`
examples/extract_invoices/  a worked pile-of-documents task, end to end
scripts/dry_run_check.py         offline wiring check for `ringer run` (mocks the model calls)
scripts/dry_run_intake_check.py  offline wiring check for `ringer intake`, same idea
```

## Verifying without spending tokens

```bash
python scripts/dry_run_check.py
python scripts/dry_run_intake_check.py
```

Both monkeypatch the model calls with deterministic fakes and assert the
retry loop, concurrency, scorecard recording, and (for the second script)
the full prompt → proposal → confirm → execute path all behave — useful
for CI or for checking a change without paying for live model calls.
