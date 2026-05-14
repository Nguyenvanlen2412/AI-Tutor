"""
server.py – FastAPI server for the AI Tutor.

Endpoints
─────────
POST  /api/chat/stream     Streaming SSE (preferred) – real-time text + audio
POST  /api/chat            Blocking fallback (waits for full response)
GET   /api/sessions        List sessions (most recent first)
GET   /api/sessions/{id}   History + summary for a session
POST  /api/sessions        Create a named session
DELETE /api/sessions/{id}  Delete a session

Start:
    uvicorn server:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from typing import Optional

import redis as redis_lib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import cfg
from graph import tutor_graph
from state import State
from Streaming import run_streaming_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Tutor", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Redis ─────────────────────────────────────────────────────────────────────
_redis = redis_lib.from_url(cfg.REDIS_URL, decode_responses=True)
_SESSION_META   = "ai_tutor:session:{sid}:meta"
_SESSION_IDX    = "ai_tutor:sessions"
_MEM_HISTORY    = "mem:history:{sid}"
_MEM_SUMMARY    = "mem:summary:{sid}"
_SESSION_TTL    = 60 * 60 * 24 * 30    # 30 days


# ─────────────────────────────────────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────────────────────────────────────

def _meta_key(sid: str) -> str: return _SESSION_META.format(sid=sid)
def _hist_key(sid: str) -> str: return _MEM_HISTORY.format(sid=sid)
def _sum_key(sid: str)  -> str: return _MEM_SUMMARY.format(sid=sid)


def _register_session(sid: str, name: str) -> dict:
    now  = time.time()
    meta = {"session_id": sid, "name": name, "created_at": now, "updated_at": now}
    pipe = _redis.pipeline()
    pipe.set(_meta_key(sid), json.dumps(meta), ex=_SESSION_TTL)
    pipe.zadd(_SESSION_IDX, {sid: now})
    pipe.execute()
    return meta


def _touch(sid: str) -> None:
    raw = _redis.get(_meta_key(sid))
    if not raw:
        return
    meta = json.loads(raw)
    meta["updated_at"] = time.time()
    _redis.set(_meta_key(sid), json.dumps(meta), ex=_SESSION_TTL)
    _redis.zadd(_SESSION_IDX, {sid: meta["updated_at"]})


def _ensure_session(sid: str) -> None:
    if not _redis.exists(_meta_key(sid)):
        _register_session(sid, f"Session {sid[:8]}")


def _build_initial_state(
    session_id: str,
    user_id: str,
    text: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
) -> State:
    common: State = {
        "output_format":      "text_and_voice",
        "session_id":         session_id,
        "user_id":            user_id,
        "regenerated_count":  0,
        "conversation_history": [],
    }
    if audio_bytes is not None:
        return {**common, "input_format": "voice", "raw_audio": audio_bytes}
    return {**common, "input_format": "text", "user_query": text or ""}


# ─────────────────────────────────────────────────────────────────────────────
# Session endpoints
# ─────────────────────────────────────────────────────────────────────────────

class CreateSessionBody(BaseModel):
    name: Optional[str]    = None
    user_id: Optional[str] = "user"


@app.post("/api/sessions")
async def create_session(body: CreateSessionBody):
    sid  = str(uuid.uuid4())
    name = body.name or f"Session {sid[:8]}"
    return {"session": _register_session(sid, name)}


@app.get("/api/sessions")
async def list_sessions(limit: int = 30):
    ids = _redis.zrevrange(_SESSION_IDX, 0, limit - 1)
    sessions = []
    for sid in ids:
        raw = _redis.get(_meta_key(sid))
        if raw:
            sessions.append(json.loads(raw))
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    raw = _redis.get(_meta_key(session_id))
    if not raw:
        raise HTTPException(404, "Session not found")
    history = json.loads(_redis.get(_hist_key(session_id)) or "[]")
    summary = _redis.get(_sum_key(session_id)) or ""
    return {"session": json.loads(raw), "history": history, "summary": summary}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    pipe = _redis.pipeline()
    for key in [_meta_key(session_id), _hist_key(session_id), _sum_key(session_id)]:
        pipe.delete(key)
    pipe.zrem(_SESSION_IDX, session_id)
    pipe.execute()
    return {"deleted": session_id}


# ─────────────────────────────────────────────────────────────────────────────
# Streaming chat  /api/chat/stream
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(
    session_id: str           = Form(...),
    user_id:    str           = Form(default="user"),
    text:       Optional[str] = Form(default=None),
    audio:      Optional[UploadFile] = File(default=None),
):
    """
    Returns a streaming SSE response.
    The client reads it via fetch() + ReadableStream (see frontend).

    Each line is:  data: {...JSON...}\\n\\n
    """
    _ensure_session(session_id)

    audio_bytes = (await audio.read()) if audio is not None else None
    initial     = _build_initial_state(session_id, user_id, text, audio_bytes)

    async def event_generator():
        try:
            async for chunk in run_streaming_pipeline(initial):
                yield chunk
        except Exception as exc:
            logger.exception("Streaming pipeline error")
            import json as _json
            yield f"data: {_json.dumps({'type':'error','message':str(exc)})}\n\n"
        finally:
            _touch(session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",        # nginx: disable proxy buffering
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Blocking chat  /api/chat  (kept for debugging / non-streaming clients)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(
    session_id: str           = Form(...),
    user_id:    str           = Form(default="user"),
    text:       Optional[str] = Form(default=None),
    audio:      Optional[UploadFile] = File(default=None),
):
    _ensure_session(session_id)
    audio_bytes = (await audio.read()) if audio is not None else None
    initial     = _build_initial_state(session_id, user_id, text, audio_bytes)

    import asyncio
    loop = asyncio.get_event_loop()
    t0   = time.monotonic()

    try:
        final: State = await loop.run_in_executor(None, tutor_graph.invoke, initial)
    except Exception as exc:
        logger.exception("Graph invocation failed")
        raise HTTPException(500, str(exc))

    audio_b64 = None
    if final.get("audio_response"):
        audio_b64 = base64.b64encode(final["audio_response"]).decode()

    _touch(session_id)
    return JSONResponse({
        "session_id":     session_id,
        "user_query":     final.get("user_query", text or ""),
        "response":       final.get("llm_response", ""),
        "audio_b64":      audio_b64,
        "sources":        final.get("sources", []),
        "latency_ms":     int((time.monotonic() - t0) * 1000),
        "error":          final.get("error_message", ""),
        "safety_blocked": final.get("input_safety_status") == "unsafe",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Static frontend
# ─────────────────────────────────────────────────────────────────────────────

if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)