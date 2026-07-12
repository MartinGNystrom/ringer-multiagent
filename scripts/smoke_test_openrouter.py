"""Direct OpenRouter plumbing check -- bypasses the planner entirely.

`ringer run ...:build_task_openrouter` lets the *planner* decide whether any
unit is worth routing to OpenRouter, and it's allowed to decide "no" (e.g.
every unit got marked needs_judge, which the tier guidance in planner.py
explicitly steers away from OpenRouter tiers). That's a planning decision,
not proof the OpenRouter call path itself works.

This script skips the planner and hands worker.run() a TaskSpec with
tier="openrouter_glm" directly, so a real call to OpenRouter happens no
matter what. Useful for confirming OPENROUTER_API_KEY / auth / the
call_json plumbing in isolation. Costs a fraction of a cent (GLM 5.2 is
$0.35/$1.10 per Mtok and this prompt is tiny).

Run: python scripts/smoke_test_openrouter.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ringer import _openrouter_client
from ringer.spec import TaskSpec
from ringer import worker


async def main() -> None:
    if not _openrouter_client.is_configured():
        print("OPENROUTER_API_KEY is not set in this environment -- nothing to test.")
        sys.exit(1)

    spec = TaskSpec(
        unit_id="smoke-test",
        instructions="Say hello and name the two colors of the French flag that are not blue.",
        input_data="(no input data needed for this smoke test)",
        output_schema={
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "colors": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["greeting", "colors"],
            "additionalProperties": False,
        },
        tier="openrouter_glm",
    )

    print("calling OpenRouter (GLM 5.2) directly, bypassing the planner...")
    result = await worker.run(spec)

    print(f"\nmodel served by: {result.model}")
    print(f"tokens:          {result.input_tokens} in / {result.output_tokens} out")
    print(f"cost:            ${result.cost:.6f}")
    print(f"output:          {result.output}")
    print("\nOpenRouter call path confirmed working.")


if __name__ == "__main__":
    asyncio.run(main())
