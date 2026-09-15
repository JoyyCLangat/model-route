"""A tiny in process event bus. The trace is the demo.

Every routing decision and every model call becomes one event. Events are
appended to runs/latest.jsonl, so there is always a record on disk, and pushed
live to anything that has subscribed, which is how the web panel updates in
real time.

replay() reads a recorded run back and re emits it with the original pauses
between events, so an offline replay looks and feels like a live run. That is
what lets someone with no API key still see exactly what happened.
"""

import json
import os
import queue
import threading
import time
from typing import Iterator

LATEST_PATH = os.path.join("runs", "latest.jsonl")

_subscribers: list = []
_lock = threading.Lock()


def emit(event: dict) -> None:
    """Record one event and push it to every live subscriber."""
    _append(event)
    with _lock:
        targets = list(_subscribers)
    for q in targets:
        q.put(event)


def subscribe() -> Iterator[dict]:
    """Yield events as they are emitted. Used by the web panel over SSE."""
    q: queue.Queue = queue.Queue()
    with _lock:
        _subscribers.append(q)
    try:
        while True:
            yield q.get()
    finally:
        with _lock:
            if q in _subscribers:
                _subscribers.remove(q)


def replay(path: str, speed: float = 1.0) -> Iterator[dict]:
    """Read a recorded JSONL run and re emit it with the original timing.

    speed scales the gaps. speed of 2 plays twice as fast. The gaps come from
    the ts field, so the rhythm of the real run is preserved. A very long gap
    is capped so a demo never stalls waiting on a recording.
    """
    events = _load(path)
    previous_ts = None
    for event in events:
        ts = event.get("ts")
        if previous_ts is not None and ts is not None and speed > 0:
            gap = (ts - previous_ts) / speed
            if gap > 0:
                time.sleep(min(gap, 5.0))
        previous_ts = ts
        emit(event)
        yield event


def _append(event: dict) -> None:
    os.makedirs(os.path.dirname(LATEST_PATH), exist_ok=True)
    with open(LATEST_PATH, "a") as handle:
        handle.write(json.dumps(event) + "\n")


def _load(path: str) -> list:
    events = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
