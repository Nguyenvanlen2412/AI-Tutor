"""
server.py – FastAPI server exposing the AI Tutor LangGraph pipeline as HTTP endpoints.

Start with:
    uvicorn server:app --host 0.0.0.0 --port 8080 --reload

Endpoints:
    POST /api/chat          – text or voice turn (multipart form)
    GET  /api/sessions      – list all active sessions from Redis
    GET  /api/sessions/{id} – conversation history for a session
    POST /api/sessions      – create a new named session
    DELETE /api/sessions/{id} – delete a session from Redis
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from typing import Optional

import redis as redis_lib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import cfg
from graph import tutor_graph
from state import State

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Tutor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Redis client (for session management metadata) ────────────────────────────
_redis = redis_lib.from_url(cfg.REDIS_URL, decode_responses=True)
SESSION_META_KEY = "ai_tutor:session:{sid}:meta"
SESSION_INDEX_KEY = "ai_tutor:sessions"
SESSION_TTL = 60 * 60 * 24 * 30   # 30 days


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mem_history_key(sid: str) -> str:
    return f"mem:history:{sid}"

def _mem_summary_key(sid: str) -> str:
    return f"mem:summary:{sid}"

def _session_meta_key(sid: str) -> str:
    return SESSION_META_KEY.format(sid=sid)

def _get_session_history(session_id: str) -> list[dict]:
    raw = _redis.get(_mem_history_key(session_id))
    return json.loads(raw) if raw else []

def _get_session_summary(session_id: str) -> str:
    return _redis.get(_mem_summary_key(session_id)) or ""

def _register_session(session_id: str, name: str) -> dict:
    now = time.time()
    meta = {
        "session_id": session_id,
        "name": name,
        "created_at": now,
        "updated_at": now,
    }
    pipe = _redis.pipeline()
    pipe.set(_session_meta_key(session_id), json.dumps(meta), ex=SESSION_TTL)
    pipe.zadd(SESSION_INDEX_KEY, {session_id: now})
    pipe.execute()
    return meta

def _touch_session(session_id: str) -> None:
    raw = _redis.get(_session_meta_key(session_id))
    if raw:
        meta = json.loads(raw)
        meta["updated_at"] = time.time()
        _redis.set(_session_meta_key(session_id), json.dumps(meta), ex=SESSION_TTL)
        _redis.zadd(SESSION_INDEX_KEY, {session_id: time.time()})


# ─────────────────────────────────────────────────────────────────────────────
# Session endpoints
# ─────────────────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    name: Optional[str] = None
    user_id: Optional[str] = "user"


@app.post("/api/sessions")
async def create_session(body: CreateSessionRequest):
    session_id = str(uuid.uuid4())
    name = body.name or f"Session {session_id[:8]}"
    meta = _register_session(session_id, name)
    return {"session": meta}


@app.get("/api/sessions")
async def list_sessions(limit: int = 20):
    """Return sessions sorted by most-recently-updated."""
    ids = _redis.zrevrange(SESSION_INDEX_KEY, 0, limit - 1)
    sessions = []
    for sid in ids:
        raw = _redis.get(_session_meta_key(sid))
        if raw:
            sessions.append(json.loads(raw))
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    raw = _redis.get(_session_meta_key(session_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Session not found")
    meta = json.loads(raw)
    history = _get_session_history(session_id)
    summary = _get_session_summary(session_id)
    return {"session": meta, "history": history, "summary": summary}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    pipe = _redis.pipeline()
    pipe.delete(_session_meta_key(session_id))
    pipe.delete(_mem_history_key(session_id))
    pipe.delete(_mem_summary_key(session_id))
    pipe.zrem(SESSION_INDEX_KEY, session_id)
    pipe.execute()
    return {"deleted": session_id}


# ─────────────────────────────────────────────────────────────────────────────
# Chat endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(
    session_id: str = Form(...),
    user_id: str = Form(default="user"),
    text: Optional[str] = Form(default=None),
    audio: Optional[UploadFile] = File(default=None),
):
    """
    Accepts either:
      - text  (form field)       → text input mode
      - audio (multipart file)   → voice input mode

    Always returns both text response and base64-encoded WAV audio.
    """
    # ── Ensure session is registered ──────────────────────────────────────────
    if not _redis.exists(_session_meta_key(session_id)):
        _register_session(session_id, f"Session {session_id[:8]}")

    # ── Build initial state ───────────────────────────────────────────────────
    if audio is not None:
        audio_bytes = await audio.read()
        initial: State = {
            "input_format":       "voice",
            "output_format":      "text_and_voice",
            "raw_audio":          audio_bytes,
            "session_id":         session_id,
            "user_id":            user_id,
            "regenerated_count":  0,
            "conversation_history": [],
        }
    elif text:
        initial: State = {
            "input_format":       "text",
            "output_format":      "text_and_voice",
            "user_query":         text,
            "session_id":         session_id,
            "user_id":            user_id,
            "regenerated_count":  0,
            "conversation_history": [],
        }
    else:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'audio'.")

    # ── Run LangGraph ─────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        final: State = tutor_graph.invoke(initial)
    except Exception as exc:
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=str(exc))
    latency_ms = int((time.monotonic() - t0) * 1000)

    # ── Encode audio to base64 ────────────────────────────────────────────────
    audio_b64: Optional[str] = None
    if final.get("audio_response"):
        audio_b64 = base64.b64encode(final["audio_response"]).decode()

    _touch_session(session_id)

    return JSONResponse({
        "session_id":    session_id,
        "user_query":    final.get("user_query", text or ""),
        "response":      final.get("llm_response", ""),
        "audio_b64":     audio_b64,          # WAV, base64-encoded
        "sources":       final.get("sources", []),
        "is_cache_hit":  final.get("is_cache_hit", False),
        "latency_ms":    latency_ms,
        "error":         final.get("error_message", ""),
        "safety_blocked": final.get("input_safety_status") == "unsafe",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Serve static frontend (place index.html in ./frontend/)
# ─────────────────────────────────────────────────────────────────────────────
import os
if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
