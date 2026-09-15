"""The single provider gateway.

Every call to a model goes through this one function. No other file in the
repo may open an HTTP connection to a provider. Keeping all provider traffic
in one place means the cost accounting, the error handling and the attribution
headers live in one place, and the rest of the code never thinks about HTTP.

The most important rule here: call() never raises. On any problem it returns a
dict with ok set to False and a plain error string. The router and the agents
depend on this, because a failed call is a normal event that triggers a retry
on a different model, not a crash.
"""

import os
import time

import httpx
from dotenv import load_dotenv

# Read .env once when the module loads so a local key is picked up with no
# extra setup. Real environment variables still win over the file.
load_dotenv()

# Every request goes to this one endpoint. This is the only provider URL in
# the whole repo.
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# OpenRouter asks callers to identify themselves with these two headers. They
# are used for attribution on their side and are not secret.
_REFERER = "https://github.com/JoyyCLangat/model-route"
_TITLE = "model-route demo"


def call(model: str, prompt: str, system: str = "",
         timeout: float = 60.0, max_tokens: int = 2048) -> dict:
    """Send one prompt to one model and return a plain dict.

    On success the dict has ok True plus the text, token counts, cost in
    dollars and wall clock latency. On any failure it has ok False and an
    error string. It never raises.
    """
    start = time.perf_counter()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        # Missing key is the most common first run problem, so name the fix.
        return _fail(model, start,
                     "OPENROUTER_API_KEY is not set. Copy .env.example to "
                     ".env and add your key, or use --replay to run from a "
                     "recorded trace with no key at all.")

    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": _REFERER,
        "X-Title": _TITLE,
    }
    body = {
        "model": model,
        "messages": _messages(system, prompt),
        "max_tokens": max_tokens,
        # Ask for real usage and cost in the response, so the price comes from
        # the provider and never from a number typed into this repo.
        "usage": {"include": True},
    }

    try:
        response = httpx.post(ENDPOINT, headers=headers, json=body,
                              timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException:
        return _fail(model, start, f"timed out after {timeout:.0f}s")
    except httpx.HTTPStatusError as exc:
        detail = _http_detail(exc)
        return _fail(model, start,
                     f"http {exc.response.status_code}: {detail}")
    except httpx.HTTPError as exc:
        return _fail(model, start, f"network error: {exc}")
    except ValueError:
        # response.json() raised, so the body was not JSON.
        return _fail(model, start, "provider did not return JSON")

    # Pull the fields out defensively. A provider can return a 200 whose body
    # is missing the parts we expect, so treat that as a failure too.
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        err = data.get("error") if isinstance(data, dict) else None
        return _fail(model, start, f"unexpected response shape: {err or data}")

    usage = data.get("usage") or {}
    return {
        "ok": True,
        "model": data.get("model", model),
        "text": text or "",
        "in_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "out_tokens": int(usage.get("completion_tokens", 0) or 0),
        # OpenRouter returns cost in dollars when usage.include is true.
        "cost": float(usage.get("cost", 0.0) or 0.0),
        "latency": time.perf_counter() - start,
    }


def _messages(system: str, prompt: str) -> list:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _fail(model: str, start: float, error: str) -> dict:
    return {"ok": False, "model": model, "cost": 0.0,
            "latency": time.perf_counter() - start, "error": error}


def _http_detail(exc: httpx.HTTPStatusError) -> str:
    # Providers usually put a clear message in the error body. Show it.
    try:
        body = exc.response.json()
        if isinstance(body, dict) and "error" in body:
            err = body["error"]
            if isinstance(err, dict):
                return str(err.get("message", err))
            return str(err)
    except ValueError:
        pass
    return exc.response.text[:200]
