"""
server.py – FastAPI server exposing the AI Tutor LangGraph pipeline as HTTP endpoints.

Start with:
    uvicorn server:app --host 0.0.0.0 --port 8080 --reload

Endpoints:
    POST /api/chat          – text or voice turn (multipart form, batched response)
    POST /api/chat/stream   – text or voice turn (SSE, sentence-by-sentence streaming)
    GET  /api/sessions      – list all active sessions from Redis
    GET  /api/sessions/{id} – conversation history for a session
    POST /api/sessions      – create a new named session
    DELETE /api/sessions/{id} – delete a session from Redis

── Streaming design (/api/chat/stream) ─────────────────────────────────────────
The streaming endpoint runs the same input pipeline (STT → safety → retrieval)
then enters a two-coroutine pipeline for sentence-level TTS:

  LLM producer coroutine:
    Streams tokens from the LLM via LLMService.stream_chat().
    Accumulates tokens and splits on sentence boundaries.
    Pushes complete sentences to sentence_queue.

  TTS worker coroutine:
    Pulls sentences from sentence_queue one at a time.
    Calls TTSService.synthesize_sentence() for each sentence.
    Pushes (text, audio_b64) pairs to result_queue.

  SSE writer (main coroutine):
    Reads result_queue and emits one SSE "sentence" event per item.

Because the LLM producer and TTS worker run concurrently, TTS for sentence N
overlaps with the LLM generating tokens for sentence N+1 — audio is ready
with minimal additional delay after each sentence is complete.

SSE event types:
  {"type": "transcript",    "text": str}           voice only — user transcript
  {"type": "sentence",      "index": int,
                             "text": str,
                             "audio_b64": str|null} one per sentence
  {"type": "done",          "full_response": str,
                             "sources": [...],
                             "is_cache_hit": bool,
                             "latency_ms": int,
                             "safety_blocked": bool}
  {"type": "safety_blocked","message": str}         input blocked before LLM
  {"type": "error",         "message": str}         unexpected exception
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import uuid
from typing import AsyncGenerator, Optional

import redis as redis_lib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from config import cfg
from nodes import (
    get_user_input,
    speech_to_text,
    check_input_vulnerability,
    handle_input_vulnerability,
    retrieve_context,
    save_context,
    SYSTEM_PROMPT,
)
from graph import tutor_graph
from state import State
from services import get_llm, get_tts, get_safety

load_dotenv()

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


# ── Sentence splitting ────────────────────────────────────────────────────────
# Matches a sentence boundary: one of . ! ? followed by one or more spaces,
# OR two or more consecutive newlines (paragraph break).
# The lookbehind ensures we only split after actual punctuation, not mid-word.
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+|(?<=\n)\n+')

def _split_sentences(text: str) -> list[str]:
    """
    Split *text* on sentence boundaries.
    In streaming mode the last element of the returned list may be an
    incomplete sentence — callers should hold it as a buffer and only
    flush it after the LLM stream ends.
    """
    parts = _SENT_BOUNDARY.split(text.strip())
    return [p for p in parts if p] or [text]


# ── Message builder (mirrors create_response in nodes.py) ─────────────────────
def _build_llm_messages(state: State) -> list[dict]:
    """
    Construct the messages list for the LLM exactly as create_response does,
    so the streaming endpoint produces identical prompts to the batched path.
    """
    reranked    = state.get("reranked_context", [])
    history     = state.get("conversation_history", [])
    summary     = state.get("summarized_memory", "")
    query       = state.get("user_query", "")
    regen_count = state.get("regenerated_count", 0)

    ctx_parts: list[str] = []
    if summary:
        ctx_parts.append(f"[Tóm tắt cuộc hội thoại trước]\n{summary}")
    if reranked:
        ctx_str = "\n\n".join(
            f"[Tài liệu {i+1}] {p}" for i, p in enumerate(reranked)
        )
        ctx_parts.append(f"[Ngữ cảnh từ tài liệu]\n{ctx_str}")
    else:
        ctx_parts.append(
            "[Lưu ý: Không tìm thấy tài liệu liên quan trong kho dữ liệu. "
            "Trả lời dựa trên kiến thức chung, nhưng nêu rõ rằng thông tin này "
            "không có trong tài liệu được cung cấp.]"
        )

    system_content = SYSTEM_PROMPT
    if ctx_parts:
        system_content += "\n\n" + "\n\n".join(ctx_parts)
    if regen_count > 0:
        system_content += (
            "\n\n[QUAN TRỌNG] Câu trả lời trước đã vi phạm chính sách an toàn. "
            "Hãy cung cấp một câu trả lời hoàn toàn an toàn, phù hợp và hữu ích."
        )

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history[-cfg.MEMORY_WINDOW:])
    messages.append({"role": "user", "content": query})
    return messages


def _sse(data: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


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
# Batched chat endpoint (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(
    background_tasks: BackgroundTasks,
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
    if not _redis.exists(_session_meta_key(session_id)):
        _register_session(session_id, f"Session {session_id[:8]}")

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

    t0 = time.monotonic()
    try:
        final: State = await tutor_graph.ainvoke(initial)
    except Exception as exc:
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=str(exc))
    latency_ms = int((time.monotonic() - t0) * 1000)

    background_tasks.add_task(save_context, final)

    audio_b64: Optional[str] = None
    if final.get("audio_response"):
        audio_b64 = base64.b64encode(final["audio_response"]).decode()

    _touch_session(session_id)

    return JSONResponse({
        "session_id":    session_id,
        "user_query":    final.get("user_query", text or ""),
        "response":      final.get("llm_response", ""),
        "audio_b64":     audio_b64,
        "sources":       final.get("sources", []),
        "is_cache_hit":  final.get("is_cache_hit", False),
        "latency_ms":    latency_ms,
        "error":         final.get("error_message", ""),
        "safety_blocked": final.get("input_safety_status") == "unsafe",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Streaming chat endpoint  [NEW]
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    user_id: str = Form(default="user"),
    text: Optional[str] = Form(default=None),
    audio: Optional[UploadFile] = File(default=None),
):
    """
    Streaming variant of /api/chat.

    Runs the same input pipeline (STT → input safety → context retrieval),
    then streams the LLM response sentence by sentence.  Each complete
    sentence is synthesised to speech immediately and sent as a Server-Sent
    Event before the next sentence is generated.

    The LLM producer and TTS worker run as concurrent asyncio tasks so that
    TTS for sentence N overlaps with the LLM generating sentence N+1.
    """
    if not text and audio is None:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'audio'.")

    if not _redis.exists(_session_meta_key(session_id)):
        _register_session(session_id, f"Session {session_id[:8]}")

    # Read audio bytes before entering the generator (UploadFile is not safe
    # to read inside a background generator after the request scope closes).
    audio_bytes: Optional[bytes] = None
    if audio is not None:
        audio_bytes = await audio.read()

    async def event_generator() -> AsyncGenerator[str, None]:
        t0 = time.monotonic()

        try:
            # ── Build initial state ──────────────────────────────────────────
            if audio_bytes is not None:
                initial: State = {
                    "input_format":         "voice",
                    "output_format":        "text_and_voice",
                    "raw_audio":            audio_bytes,
                    "session_id":           session_id,
                    "user_id":              user_id,
                    "regenerated_count":    0,
                    "conversation_history": [],
                }
            else:
                initial: State = {
                    "input_format":         "text",
                    "output_format":        "text_and_voice",
                    "user_query":           text,
                    "session_id":           session_id,
                    "user_id":              user_id,
                    "regenerated_count":    0,
                    "conversation_history": [],
                }

            # ── Input pipeline ───────────────────────────────────────────────
            state = await get_user_input(initial)

            if state.get("input_format") == "voice":
                state = await speech_to_text(state)
                if state.get("error_message"):
                    yield _sse({"type": "error", "message": state["error_message"]})
                    return
                # Send transcript so the client can update the user bubble.
                yield _sse({"type": "transcript", "text": state.get("user_query", "")})

            state = await check_input_vulnerability(state)

            if state.get("input_safety_status") == "unsafe":
                yield _sse({"type": "safety_blocked", "message": cfg.SAFE_FALLBACK_MESSAGE})
                return

            state = await retrieve_context(state)

            # ── Sentence-streaming pipeline ──────────────────────────────────
            tts = get_tts()
            full_response_parts: list[str] = []

            if state.get("is_cache_hit"):
                # ── Cache hit: TTS the pre-formed response sentence-by-sentence
                full_response = state.get("llm_response", "")
                full_response_parts = [full_response]
                sentences = _split_sentences(full_response)
                for idx, sent in enumerate(sentences):
                    sent = sent.strip()
                    if not sent:
                        continue
                    try:
                        audio_b = await tts.synthesize_sentence(sent)
                        a64 = base64.b64encode(audio_b).decode() if audio_b else None
                    except Exception as exc:
                        logger.warning(f"[stream] TTS failed for cached sentence: {exc}")
                        a64 = None
                    yield _sse({"type": "sentence", "index": idx, "text": sent, "audio_b64": a64})

            else:
                # ── Cache miss: LLM producer → TTS worker → SSE writer ────────
                messages = _build_llm_messages(state)
                llm = get_llm()

                # Queues use None as sentinel to signal end-of-stream.
                sentence_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
                result_queue: asyncio.Queue[Optional[tuple]] = asyncio.Queue()

                async def llm_producer() -> None:
                    """Stream tokens, split into sentences, push to sentence_queue."""
                    buf = ""
                    try:
                        async for token in llm.stream_chat(messages):
                            buf += token
                            full_response_parts.append(token)
                            parts = _split_sentences(buf)
                            # All parts except the last are complete sentences.
                            for sent in parts[:-1]:
                                sent = sent.strip()
                                if sent:
                                    await sentence_queue.put(sent)
                            buf = parts[-1]  # hold incomplete tail
                        # Flush the final (possibly unterminated) sentence.
                        tail = buf.strip()
                        if tail:
                            await sentence_queue.put(tail)
                    except Exception as exc:
                        logger.error(f"[stream] LLM producer error: {exc}")
                        full_response_parts.append(
                            "\n\nXin lỗi, đã xảy ra lỗi khi tạo câu trả lời."
                        )
                    finally:
                        await sentence_queue.put(None)  # sentinel

                async def tts_worker() -> None:
                    """Pull sentences, synthesise, push (text, audio_b64) to result_queue."""
                    while True:
                        sent = await sentence_queue.get()
                        if sent is None:
                            await result_queue.put(None)  # propagate sentinel
                            break
                        try:
                            audio_b = await tts.synthesize_sentence(sent)
                            a64 = base64.b64encode(audio_b).decode() if audio_b else None
                        except Exception as exc:
                            logger.warning(f"[stream] TTS worker error: {exc}")
                            a64 = None
                        await result_queue.put((sent, a64))

                # Start both tasks concurrently so TTS overlaps with LLM streaming.
                producer_task = asyncio.create_task(llm_producer())
                worker_task   = asyncio.create_task(tts_worker())

                idx = 0
                while True:
                    item = await result_queue.get()
                    if item is None:
                        break
                    sent, a64 = item
                    yield _sse({"type": "sentence", "index": idx, "text": sent, "audio_b64": a64})
                    idx += 1

                # Ensure both tasks are fully done before continuing.
                await producer_task
                await worker_task

            # ── Output safety check (non-blocking: done after all SSE sent) ──
            full_response = "".join(full_response_parts)
            safety = get_safety()
            is_safe, _ = await safety.check(full_response, role="Agent")

            latency_ms = int((time.monotonic() - t0) * 1000)
            yield _sse({
                "type":           "done",
                "full_response":  full_response,
                "sources":        state.get("sources", []),
                "is_cache_hit":   state.get("is_cache_hit", False),
                "latency_ms":     latency_ms,
                "safety_blocked": not is_safe,
            })

            # ── Persist context in background ─────────────────────────────────
            final_state = {**state, "llm_response": full_response}
            background_tasks.add_task(save_context, final_state)
            _touch_session(session_id)

        except Exception as exc:
            logger.exception("[stream] Unhandled error in event_generator")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection":       "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Serve static frontend (place index.html in ./frontend/)
# ─────────────────────────────────────────────────────────────────────────────
import os
if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)