"""The live model catalog.

Prices and context limits are not written into this repo. They are fetched
from OpenRouter at run time, because they change often and a number baked into
the code goes stale without anyone noticing. The only opinions this repo holds
are the capability scores in capabilities.yaml, and even those are keyed by
family so they survive a model version bump.

The response is cached on disk for a day, so a live demo is not at the mercy
of one network call. Pass force=True to refetch.
"""

import json
import os
import time

import httpx

MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_PATH = os.path.join(".cache", "catalog.json")
CACHE_TTL = 24 * 60 * 60  # one day, in seconds


def fetch(force: bool = False) -> list[dict]:
    """Return the normalised catalog as a list of dicts.

    Uses the day old cache when it can. Falls back to a stale cache if the
    network is down, because a slightly old price list beats no demo at all.
    """
    if not force:
        cached = _read_cache(max_age=CACHE_TTL)
        if cached is not None:
            return _normalise(cached)

    try:
        response = httpx.get(MODELS_URL, timeout=30.0)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        stale = _read_cache(max_age=None)
        if stale is not None:
            print(f"catalog: network failed ({exc}), using the cached copy")
            return _normalise(stale)
        raise RuntimeError(
            f"could not fetch the model catalog and there is no cache: {exc}"
        ) from exc

    _write_cache(payload)
    return _normalise(payload)


def _normalise(payload: dict) -> list[dict]:
    """Turn the raw OpenRouter response into the small shape the repo uses."""
    models = []
    for entry in payload.get("data", []):
        pricing = entry.get("pricing") or {}
        in_price = _per_million(pricing.get("prompt"))
        out_price = _per_million(pricing.get("completion"))
        slug = entry.get("id", "")
        models.append({
            "id": slug,
            "ctx": _context(entry),
            "in_price": in_price,
            "out_price": out_price,
            "modalities": _modalities(entry),
            # Free either by the :free suffix convention or by a zero price.
            "is_free": slug.endswith(":free")
                       or (in_price == 0 and out_price == 0),
        })
    return models


def _per_million(raw) -> float:
    # OpenRouter prices are dollars per single token, sent as a string. People
    # think in dollars per million tokens, so convert once, here.
    try:
        return float(raw) * 1_000_000
    except (TypeError, ValueError):
        return 0.0


def _context(entry: dict) -> int:
    ctx = entry.get("context_length")
    if not ctx:
        ctx = (entry.get("top_provider") or {}).get("context_length")
    try:
        return int(ctx or 0)
    except (TypeError, ValueError):
        return 0


def _modalities(entry: dict) -> list:
    arch = entry.get("architecture") or {}
    mods = arch.get("input_modalities")
    if isinstance(mods, list) and mods:
        return list(mods)
    # Older entries only carry a modality string like "text+image->text".
    modality = arch.get("modality") or "text"
    left = modality.split("->")[0]
    return [p.strip() for p in left.replace("+", " ").split() if p.strip()]


def _read_cache(max_age):
    try:
        with open(CACHE_PATH) as handle:
            blob = json.load(handle)
    except (OSError, ValueError):
        return None
    if max_age is not None and time.time() - blob.get("fetched_at", 0) > max_age:
        return None
    return blob.get("response")


def _write_cache(payload: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as handle:
        json.dump({"fetched_at": time.time(), "response": payload}, handle)


def _print_table() -> None:
    # Step 1 of the talk: show what the key can actually reach, with live
    # prices. We show three families here to keep the screen readable.
    from rich.console import Console
    from rich.table import Table

    families = ("claude", "gemini", "kimi")
    catalog = fetch()
    rows = [m for m in catalog if any(f in m["id"].lower() for f in families)]
    rows.sort(key=lambda m: m["id"])

    table = Table(title="models on this key, by family (live prices)")
    table.add_column("id")
    table.add_column("ctx", justify="right")
    table.add_column("$ in / Mtok", justify="right")
    table.add_column("$ out / Mtok", justify="right")
    table.add_column("free")
    for m in rows:
        table.add_row(m["id"], f"{m['ctx']:,}", f"{m['in_price']:.3f}",
                      f"{m['out_price']:.3f}", "yes" if m["is_free"] else "")
    Console().print(table)
    print(f"{len(rows)} models shown, {len(catalog)} total in the catalog")


if __name__ == "__main__":
    _print_table()
