"""FastAPI backend for the verification UI (CopilotKit).

Endpoints
---------
- ``POST /api/analyze``         submit code/diff -> structured report (sync by default,
                                or background with ``wait=false`` -> returns a job id)
- ``GET  /api/jobs/{id}``       current status, or the full report once done
- ``GET  /api/jobs/{id}/events``replayable list of stage events
- ``GET  /api/jobs/{id}/stream``Server-Sent Events: live stage-by-stage fan-out
- ``GET  /health``              service + Redis + Weave status

The frontend reads its data from here and Redis — not from Weave (Weave is the trace
recorder; the report carries a ``weave_trace_url`` to link out to it).

Run: ``uv run uvicorn verification_agents.api:app --reload``
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from verification_agents.specialists.redis_store import JobStore
from verification_agents.specialists.verify import verify
from verification_agents.tools import parse_diff as _parse_diff

load_dotenv()

_MODEL = os.environ.get("VERIFY_MODEL", "gpt-4o-mini")
_weave_enabled = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _weave_enabled
    if os.environ.get("WANDB_API_KEY"):
        try:
            import weave

            weave.init(os.environ.get("WEAVE_PROJECT", "astrio/verification-agents"))
            _weave_enabled = True
        except Exception as exc:  # pragma: no cover
            print(f"[api] weave disabled: {exc}")
    yield


app = FastAPI(title="Formal Verification Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    repo_url: str | None = None
    pr_url: str | None = None
    code: str | None = None
    diff: str | None = None
    selected_file: str | None = None
    target_property: str | None = None
    wait: bool = True  # run synchronously and return the full report


def _code_to_diff(code: str, filename: str) -> str:
    lines = code.splitlines() or [""]
    body = "\n".join("+" + ln for ln in lines)
    return (
        f"diff --git a/{filename} b/{filename}\nnew file mode 100644\n"
        f"--- /dev/null\n+++ b/{filename}\n@@ -0,0 +1,{len(lines)} @@\n{body}\n"
    )


def _build_analysis(req: AnalyzeRequest):
    if req.diff:
        return _parse_diff.run(req.diff)
    if req.code:
        return _parse_diff.run(_code_to_diff(req.code, req.selected_file or "pasted.py"))
    raise HTTPException(
        status_code=400,
        detail="provide `code` or `diff` (repo_url/pr_url fetching is not implemented yet)",
    )


def _run_verify(analysis, job_id: str):
    kwargs = dict(api_key=os.environ.get("OPENAI_API_KEY"), model=_MODEL,
                  job_id=job_id, z3_timeout_ms=10_000)
    if _weave_enabled and hasattr(verify, "call"):
        report, call = verify.call(analysis, **kwargs)
        report.weave_trace_url = getattr(call, "ui_url", "") or ""
    else:
        report = verify(analysis, **kwargs)
    JobStore(job_id).set_context(report.model_dump())  # persist incl. trace url
    return report


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "redis": JobStore("healthcheck").backend,
        "weave": _weave_enabled,
        "model": _MODEL,
        "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    analysis = _build_analysis(req)
    job_id = uuid.uuid4().hex[:12]
    job = JobStore(job_id)
    job.set_status("queued")

    if req.wait:
        report = _run_verify(analysis, job_id)
        return report.model_dump()

    threading.Thread(target=_run_verify, args=(analysis, job_id), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "stream": f"/api/jobs/{job_id}/stream"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    store = JobStore(job_id)
    ctx = store.get_context()
    if ctx:
        return ctx                                   # finished -> full report
    status = store.get_status()
    if status:
        return {"job_id": job_id, "status": status}  # still running
    raise HTTPException(status_code=404, detail="unknown job")


@app.get("/api/jobs/{job_id}/events")
def get_events(job_id: str) -> dict:
    store = JobStore(job_id)
    events = store.events()
    if not events and not store.get_status():
        raise HTTPException(status_code=404, detail="unknown job")
    return {"job_id": job_id, "events": events}


@app.get("/api/jobs/{job_id}/stream")
async def stream(job_id: str) -> StreamingResponse:
    async def gen():
        store = JobStore(job_id)
        sent, waited = 0, 0.0
        while waited < 180:
            events = store.events()
            for e in events[sent:]:
                yield f"data: {json.dumps(e)}\n\n"
            sent = len(events)
            if any(e.get("stage") == "done" for e in events):
                break
            await asyncio.sleep(0.25)
            waited += 0.25
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
