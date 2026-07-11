"""OpenRouter call plumbing for open-weight worker models (GLM 5.2, Kimi K2).

OpenRouter speaks the OpenAI chat-completions shape, not the Anthropic
Messages API, so this is a separate small client rather than a branch
inside _client.py. Structured outputs are far less consistently enforced
across third-party open-weight models than Anthropic's `output_config
.format`, so this deliberately does *not* raise on a JSON parse failure --
it hands the raw text back so the mechanical checker (checker.py) rejects
it like any other bad output, and the orchestrator's retry loop (with the
parse failure appended as failure context) gets a chance to fix it. That's
the harness's actual answer to "these models are less reliable": more
attempts, paired with the same cheap check everything else goes through.
"""

from __future__ import annotations

import json
import os

import httpx

from .models import estimate_cost

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterAuthError(RuntimeError):
    """Raised when OPENROUTER_API_KEY is missing."""


class CallResult:
    __slots__ = ("parsed", "input_tokens", "output_tokens", "cost", "served_by", "parse_error")

    def __init__(
        self,
        parsed,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        served_by: str,
        parse_error: str | None = None,
    ):
        self.parsed = parsed
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost = cost
        self.served_by = served_by
        self.parse_error = parse_error


def _headers() -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterAuthError(
            "OPENROUTER_API_KEY is not set -- required to route worker calls to OpenRouter models"
        )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Optional but recommended by OpenRouter for attributing traffic; harmless to omit.
    if referer := os.environ.get("RINGER_APP_URL"):
        headers["HTTP-Referer"] = referer
    if title := os.environ.get("RINGER_APP_NAME"):
        headers["X-Title"] = title
    return headers


async def call_json(
    *,
    model: str,
    system: str,
    user_content: str,
    schema: dict,
    max_tokens: int = 8000,
) -> CallResult:
    """Call an OpenRouter model, asking for JSON matching `schema`.

    Unlike ringer._client.call_json, this never raises on a malformed
    response -- see the module docstring for why. A caller (worker.py)
    that wants schema-shaped output should feed CallResult.parsed straight
    to the checker either way; a parse failure just won't validate.
    """
    schema_hint = (
        f"{system}\n\nRespond with a single JSON object only -- no prose, no markdown "
        f"fences -- matching this JSON schema exactly:\n{json.dumps(schema)}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": schema_hint},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(_OPENROUTER_URL, headers=_headers(), json=payload)
        response.raise_for_status()
        data = response.json()

    choice = data["choices"][0]
    text = choice["message"]["content"]
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    served_by = data.get("model", model)
    cost = estimate_cost(model, input_tokens, output_tokens)

    try:
        parsed = json.loads(text)
        parse_error = None
    except json.JSONDecodeError as exc:
        # Hand back the raw text under a sentinel key rather than raising --
        # schema_checker will reject it (missing required keys) with a clear
        # reason, and the orchestrator's retry loop appends the JSON error
        # too so the next attempt knows exactly what went wrong.
        parsed = {"_raw_text": text}
        parse_error = str(exc)

    return CallResult(parsed, input_tokens, output_tokens, cost, served_by, parse_error)
