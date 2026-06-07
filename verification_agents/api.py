"""FastAPI backend for the verification UI.

Endpoints
---------
- ``POST /api/analyze``         submit code/diff -> structured report (sync by default,
                                or background with ``wait=false`` -> returns a job id)
- ``GET  /api/jobs/{id}``       current status, or the full report once done
- ``GET  /api/jobs/{id}/events``replayable list of stage events
- ``WS   /api/jobs/{id}/ws``    WebSocket: live stage events (server->client) +
                                clarification responses (client->server)
- ``POST /api/tasks``           create a chat task session -> returns job_id
- ``WS   /api/tasks/{id}/ws``  persistent chat WebSocket: stream assistant tokens,
                                receive user messages over the same connection
- ``GET  /health``              service + Redis + Weave status

WebSocket message protocol (verification jobs)
----------------------------------------------
Server -> client  (JSON):
  every event from JobStore.events(), e.g.
    {"stage": "status", "status": "solving", ...}
    {"stage": "property", "concern": "array_bounds", "status": "sat", ...}
    {"stage": "clarification_needed", "question": "...", "options": [...]}
    {"stage": "done", ...}
    {"stage": "end"}   <- final sentinel

Client -> server  (JSON):
    {"type": "clarification_response", "selection": {"selected_ids": [...], "extra_notes": ""}}

WebSocket message protocol (chat tasks)
---------------------------------------
Server -> client  (JSON):
    {"stage": "ready"}                         <- session open, waiting for first message
    {"stage": "token", "text": "..."}          <- streaming assistant token
    {"stage": "message_done"}                  <- assistant turn complete
    {"stage": "error", "message": "..."}       <- unrecoverable error

Client -> server  (JSON):
    {"type": "user_message", "content": "..."}

Run: ``uv run uvicorn verification_agents.api:app --reload``
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel

from verification_agents.models import UserSelection, VerifiableProperty
from verification_agents.specialists.redis_store import JobStore
from verification_agents.specialists.verify import verify
from verification_agents.tools import parse_diff as _parse_diff

load_dotenv()

_MODEL = os.environ.get("VERIFY_MODEL", "gpt-4o-mini")
_CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o")
_weave_enabled = False

# Bridges the synchronous ask_user tool (runs in a background thread) to the
# async WebSocket handler. Keyed by job_id.
_CLARIFICATION_WAITERS: dict[str, threading.Event] = {}
_CLARIFICATION_RESPONSES: dict[str, dict] = {}

# Per-session queues for chat tasks: job_id -> asyncio.Queue of outbound WS events.
# Each queue item is a dict ready to be sent as JSON.
_CHAT_QUEUES: dict[str, asyncio.Queue[dict[str, Any]]] = {}

_CHAT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Be concise and accurate. "
    "When helping with code, prefer short focused answers over long explanations."
)


def _ws_clarification_handler(job_id: str):
    """Return a ClarificationHandler that pauses the agent until the frontend replies."""
    def handler(question: str, props: list[VerifiableProperty]) -> UserSelection:
        store = JobStore(job_id)
        store.publish({
            "stage": "clarification_needed",
            "question": question,
            "options": [p.model_dump() for p in props],
        })
        ev = threading.Event()
        _CLARIFICATION_WAITERS[job_id] = ev
        ev.wait(timeout=300)  # 5-minute window for the user to respond
        _CLARIFICATION_WAITERS.pop(job_id, None)
        raw = _CLARIFICATION_RESPONSES.pop(job_id, {"selected_ids": [], "extra_notes": "timeout"})
        return UserSelection(**raw)
    return handler


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
    return {"job_id": job_id, "status": "queued", "ws": f"/api/jobs/{job_id}/ws"}


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


@app.websocket("/api/jobs/{job_id}/ws")
async def ws_job(job_id: str, ws: WebSocket) -> None:
    await ws.accept()
    store = JobStore(job_id)
    if not store.get_status() and not store.get_context():
        await ws.send_json({"error": "unknown job"})
        await ws.close()
        return

    async def _stream() -> None:
        sent, waited = 0, 0.0
        while waited < 180:
            events = store.events()
            for e in events[sent:]:
                await ws.send_json(e)
            sent = len(events)
            if any(e.get("stage") == "done" for e in events):
                break
            await asyncio.sleep(0.25)
            waited += 0.25
        await ws.send_json({"stage": "end"})

    stream_task = asyncio.create_task(_stream())
    try:
        async for msg in ws.iter_json():
            if msg.get("type") == "clarification_response":
                _CLARIFICATION_RESPONSES[job_id] = msg["selection"]
                ev = _CLARIFICATION_WAITERS.pop(job_id, None)
                if ev:
                    ev.set()
    except WebSocketDisconnect:
        pass
    finally:
        stream_task.cancel()


# ---------------------------------------------------------------------------
# Chat task endpoints
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    prompt: str | None = None


@app.post("/api/tasks")
async def create_task(req: TaskRequest) -> dict:
    job_id = uuid.uuid4().hex[:12]
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _CHAT_QUEUES[job_id] = queue
    if req.prompt:
        await queue.put({"_user_message": req.prompt})
    return {"job_id": job_id, "status": "open", "ws": f"/api/tasks/{job_id}/ws"}


@app.websocket("/api/tasks/{job_id}/ws")
async def ws_task(job_id: str, ws: WebSocket) -> None:
    await ws.accept()

    queue = _CHAT_QUEUES.get(job_id)
    if queue is None:
        await ws.send_json({"stage": "error", "message": "unknown task"})
        await ws.close()
        return

    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    history: list[dict[str, str]] = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}]

    await ws.send_json({"stage": "ready"})

    async def _handle_user_message(content: str) -> None:
        history.append({"role": "user", "content": content})
        try:
            stream = await openai_client.chat.completions.create(
                model=_CHAT_MODEL,
                messages=history,  # type: ignore[arg-type]
                stream=True,
            )
            assistant_text = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    assistant_text += delta
                    await ws.send_json({"stage": "token", "text": delta})
            history.append({"role": "assistant", "content": assistant_text})
            await ws.send_json({"stage": "message_done"})
        except Exception as exc:
            await ws.send_json({"stage": "error", "message": str(exc)})

    async def _drain_queue() -> None:
        """Forward any pre-queued messages (e.g. initial prompt from POST body)."""
        while not queue.empty():
            item = await queue.get()
            if "_user_message" in item:
                await _handle_user_message(item["_user_message"])

    await _drain_queue()

    try:
        async for msg in ws.iter_json():
            if msg.get("type") == "user_message":
                content = str(msg.get("content", "")).strip()
                if content:
                    await _handle_user_message(content)
    except WebSocketDisconnect:
        pass
    finally:
        _CHAT_QUEUES.pop(job_id, None)
