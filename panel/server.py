"""The web panel server.

FastAPI with two routes. One serves a single HTML file. The other is a Server
Sent Events stream that forwards every trace event to the browser as it
happens. There is no build step, no npm, and no external asset, so the panel
loads on venue wifi or on no wifi at all, as long as it can reach localhost.
"""

import json
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from core import trace

app = FastAPI(title="model-route panel")

HERE = os.path.dirname(__file__)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/events")
def events():
    def stream():
        # One SSE message per event. The browser reconnects on its own if the
        # stream drops.
        for event in trace.subscribe():
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


def port_from_env() -> int:
    try:
        return int(os.environ.get("PANEL_PORT", "8000"))
    except ValueError:
        return 8000


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port_from_env())
