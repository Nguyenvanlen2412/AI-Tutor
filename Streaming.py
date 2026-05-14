"""
streaming.py – real-time LLM + TTS streaming pipeline.

Three concurrent async tasks:

  Task A  llm_producer
      Reads tokens from Ollama (NDJSON stream), pushes each token into
      token_queue AND feeds a SentenceBuffer. When a sentence boundary is
      detected the complete sentence is pushed into sentence_queue.

  Task B  tts_consumer
      Pops sentences from sentence_queue, calls TTS in a thread-executor
      (non-blocking), pushes (index, wav_bytes) into audio_queue in order.

  Main generator
      Drains token_queue and audio_queue concurrently, yielding SSE events
      to the HTTP response as fast as data arrives.

SSE event shapes:
  {"type": "user_query",    "text":       str}
  {"type": "context_ready", "sources":    list[str]}
  {"type": "text_chunk",    "text":       str}
  {"type": "audio_chunk",   "index":      int, "audio_b64": str}
  {"type": "text_end",      "full_text":  str}
  {"type": "done",          "latency_ms": int}
  {"type": "blocked",       "reason":     str}
  {"type": "error",         "message":    str}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, List, Optional

import httpx

from config import cfg
from services import get_tts
from nodes import (
    SYSTEM_PROMPT,
    get_user_input             as node_get_user_input,
    speech_to_text             as node_stt,
    check_input_vulnerability  as node_check_input,
    handle_input_vulnerability as node_handle_input,
    retrieve_context           as node_retrieve,
    save_context               as node_save,
)
from state import State

logger = logging.getLogger(__name__)

# Dedicated thread pool – keeps TTS off the default executor
_tts_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")

# Sentinel object – signals queue consumers to stop
_DONE = object()


# ─────────────────────────────────────────────────────────────────────────────
# Sentence boundary detection
# ─────────────────────────────────────────────────────────────────────────────

# Triggers after .  !  ?  …  。 ！ ？  (+ optional closing bracket/quote)
# then either: whitespace + uppercase letter, newline, or end-of-string.
_BOUNDARY = re.compile(
    r'(?<=[.!?…。！？: ])(?:["\'\)»\]]+)?'
    r'(?=[ \t]+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÙÚĂĐƠƯ]|\s*\n|\s*$)'
)
_MIN_CHARS = 15   # skip fragments shorter than this (e.g. "OK.")


class SentenceBuffer:
    """Accumulates LLM tokens; yields complete sentences on boundary detection."""

    def __init__(self) -> None:
        self._buf = ""

    def push(self, token: str) -> List[str]:
        self._buf += token
        out: List[str] = []
        while True:
            m = _BOUNDARY.search(self._buf)
            if not m:
                break
            sentence = self._buf[: m.end()].strip()
            self._buf = self._buf[m.end():]
            if len(sentence) >= _MIN_CHARS:
                out.append(sentence)
        return out

    def flush(self) -> Optional[str]:
        """Call after the stream ends to emit any remaining text."""
        text = self._buf.strip()
        self._buf = ""
        return text if len(text) >= _MIN_CHARS else None


# ─────────────────────────────────────────────────────────────────────────────
# Ollama streaming
# ─────────────────────────────────────────────────────────────────────────────

async def _stream_tokens(messages: list[dict]) -> AsyncGenerator[str, None]:
    """Yield one text token at a time from Ollama's NDJSON stream."""
    payload = {
        "model":    cfg.CORE_LLM_MODEL,
        "messages": messages,
        "stream":   True,
        "options": {
            "temperature": cfg.LLM_TEMPERATURE,
            "num_predict": cfg.LLM_MAX_TOKENS,
        },
    }
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0)
    ) as client:
        async with client.stream(
            "POST",
            f"{cfg.OLLAMA_BASE_URL}/api/chat",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.aiter_lines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                token = obj.get("message", {}).get("content", "")
                if token:
                    yield token
                if obj.get("done"):
                    break


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder (mirrors nodes.create_response)
# ─────────────────────────────────────────────────────────────────────────────

def _build_messages(state: State) -> list[dict]:
    history  = state.get("conversation_history", [])
    summary  = state.get("summarized_memory", "")
    reranked = state.get("reranked_context", [])
    query    = state.get("user_query", "")

    ctx: list[str] = []
    if summary:
        ctx.append(f"[Tóm tắt cuộc hội thoại trước]\n{summary}")
    if reranked:
        ctx.append(
            "[Ngữ cảnh từ tài liệu]\n" +
            "\n\n".join(f"[Tài liệu {i+1}] {p}" for i, p in enumerate(reranked))
        )
    system = SYSTEM_PROMPT + ("\n\n" + "\n\n".join(ctx) if ctx else "")

    msgs: list[dict] = [{"role": "system", "content": system}]
    msgs.extend(history[-cfg.MEMORY_WINDOW:])
    msgs.append({"role": "user", "content": query})
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# SSE helper
# ─────────────────────────────────────────────────────────────────────────────

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


# ─────────────────────────────────────────────────────────────────────────────
# Main streaming pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def run_streaming_pipeline(
    initial_state: State,
) -> AsyncGenerator[str, None]:
    """
    Top-level async generator – yields SSE strings.

    Phase 1  Blocking graph nodes (STT, safety, retrieval) run in executor.
    Phase 2  Three concurrent tasks: LLM tokens → sentence buffer → TTS → audio.
    Phase 3  save_context persists memory.
    """
    loop    = asyncio.get_event_loop()
    t_start = time.monotonic()

    # ── Phase 1: blocking nodes ───────────────────────────────────────────────
    try:
        # 1. Get user input
        state: State = await loop.run_in_executor(None, node_get_user_input, initial_state)
        # (Optional) Add STT here if you want voice support in streaming
        if state.get("input_format") == "voice":
            state = await loop.run_in_executor(None, node_stt, state)
        # 2. RUN THE SAFETY CHECK (This was missing)
        state = await loop.run_in_executor(None, node_check_input, state)
        # 3. Now the safety status is actually populated
        if state.get("input_safety_status") == "unsafe":
            state = await loop.run_in_executor(None, node_handle_input, state)
            fallback = state["llm_response"]
            try:
                wav = await loop.run_in_executor(_tts_pool, get_tts().synthesize, fallback)
                # Include text directly in the audio chunk
                yield _sse({"type": "audio_chunk", "index": 0, "text": fallback + " ",
                             "audio_b64": base64.b64encode(wav).decode()})
            except Exception as exc:
                yield _sse({"type": "text_chunk", "text": fallback + " "})
                logger.warning(f"TTS for blocked input failed: {exc}")
            
            yield _sse({"type": "text_end",   "full_text": fallback})
            yield _sse({"type": "blocked", "reason": state.get("blocked_reason", "")})
            yield _sse({"type": "done", "latency_ms": _ms(t_start)})
            await loop.run_in_executor(None, node_save, state)
            return

        state = await loop.run_in_executor(None, node_retrieve, state)
        yield _sse({"type": "context_ready", "sources": state.get("sources", [])})

    except Exception as exc:
        logger.exception("Phase 1 error")
        yield _sse({"type": "error", "message": str(exc)})
        return


    # ── Phase 2: two-task streaming (Synchronized) ────────────────────────────
    messages = _build_messages(state)

    sentence_q: asyncio.Queue = asyncio.Queue(maxsize=16)
    audio_q:    asyncio.Queue = asyncio.Queue()

    collected_tokens: list[str] = []

    # ── Task A: LLM producer ─────────────────────────────────────────────────
    async def llm_producer() -> None:
        buf = SentenceBuffer()
        try:
            async for token in _stream_tokens(messages):
                collected_tokens.append(token)
                for sentence in buf.push(token):
                    await sentence_q.put(sentence)
            remainder = buf.flush()
            if remainder:
                await sentence_q.put(remainder)
        except Exception as exc:
            logger.error(f"LLM producer error: {exc}")
        finally:
            await sentence_q.put(_DONE)

    # ── Task B: TTS consumer ─────────────────────────────────────────────────
    async def tts_consumer() -> None:
        idx = 0
        while True:
            sentence = await sentence_q.get()
            if sentence is _DONE:
                await audio_q.put(_DONE)
                return
            try:
                wav: bytes = await loop.run_in_executor(
                    _tts_pool, get_tts().synthesize, sentence
                )
                # Pass the sentence text along with the generated audio
                await audio_q.put((idx, sentence, wav))
                logger.debug(f"TTS chunk {idx}: '{sentence[:50]}' → {len(wav)} B")
            except Exception as exc:
                logger.warning(f"TTS chunk {idx} failed: {exc}")
                await audio_q.put((idx, sentence, b""))
            finally:
                idx += 1

    prod_task = asyncio.create_task(llm_producer())
    cons_task = asyncio.create_task(tts_consumer())

    # ── Main drain loop ───────────────────────────────────────────────────────
    audio_done = False
    audio_idx  = 0

    while not audio_done:
        try:
            # Drain audio chunks (which now contain the text too)
            item = audio_q.get_nowait()
        except asyncio.QueueEmpty:
            # Nothing ready yet, yield the event loop briefly
            await asyncio.sleep(0.012)
            continue

        if item is _DONE:
            audio_done = True
            break
        
        _, sentence, wav = item
        
        # Merge the text into the audio_chunk payload. Fallback to text_chunk if TTS failed.
        if wav:
            yield _sse({
                "type":      "audio_chunk",
                "index":     audio_idx,
                "text":      sentence + " ",
                "audio_b64": base64.b64encode(wav).decode(),
            })
            audio_idx += 1
        else:
             yield _sse({"type": "text_chunk", "text": sentence + " "})
    await prod_task
    await cons_task

    full_text = "".join(collected_tokens)
    state["llm_response"]         = full_text
    state["output_safety_status"] = "safe"

    yield _sse({"type": "text_end",  "full_text": full_text})
    yield _sse({"type": "done",      "latency_ms": _ms(t_start)})
    # ── Phase 3: persist ─────────────────────────────────────────────────────
    try:
        await loop.run_in_executor(None, node_save, state)
    except Exception as exc:
        logger.warning(f"save_context error: {exc}")