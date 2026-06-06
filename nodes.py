

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from typing import List

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from state import State
from config import cfg
from services import (
    get_vad,
    get_stt,
    get_tts,
    get_embedder,
    get_reranker,
    get_safety,
    get_llm,
    get_vector_store,
    get_memory,
    get_semantic_cache,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an intelligent, friendly, and patient AI tutor.

Your task is to help students understand concepts, solve problems, and answer questions

clearly, accurately, and in a way that encourages critical thinking.

Instructions:

- Answer in the language the student uses (English).

- If context is provided in the material, prioritize using that information.

- Cite sources if necessary.

- Do not add emojis.

- Do not give external links, but you can suggest topics or keywords for the student to research on their own.

- If unsure, be upfront and suggest the student consult additional resources.

- Encourage independent thinking by asking appropriate feedback questions.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. get_user_input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def get_user_input(state: State) -> State:
    """Capture the initial payload; assign session/user IDs if absent."""
    updates: State = {}

    if not state.get("session_id"):
        updates["session_id"] = str(uuid.uuid4())
    if not state.get("user_id"):
        updates["user_id"] = "anonymous"
    if not state.get("input_format"):
        updates["input_format"] = "text"
    if not state.get("output_format"):
        updates["output_format"] = state.get("output_format", "text_and_voice")

    updates.update(
        {
            "input_safety_status":  "pending",
            "output_safety_status": "pending",
            "regenerated_count":    state.get("regenerated_count", 0),
            "error_message":        "",
            "blocked_reason":       "",
            "vad_detected":         False,
            "transcript_confidence": 0.0,
            "is_cache_hit":         False,
            "query_embedding":      None,   # populated in retrieve_context
        }
    )

    logger.info(
        f"[get_user_input] session={state.get('session_id', updates.get('session_id'))} "
        f"format={state.get('input_format', updates.get('input_format'))}"
    )
    return {**state, **updates}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. speech_to_text  (only reached when input_format == "voice")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def speech_to_text(state: State) -> State:
    """
    1. Write audio bytes to ONE temp file (shared between VAD and STT).
    2. Run Silero VAD via detect_from_path.
    3. If speech detected, transcribe with Whisper via transcribe_from_path.

    [PERF] The original code wrote a separate temp file in VAD.detect() AND
    again in STTService.transcribe(), causing two identical disk writes per
    voice request. Now a single temp file is created here and passed to both.
    """
    audio_bytes: bytes = state.get("raw_audio", b"")
    if not audio_bytes:
        logger.warning("[speech_to_text] No audio bytes found.")
        return {**state, "error_message": "No audio provided.", "vad_detected": False}

    # Write the shared temp file once
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        # ── VAD ───────────────────────────────────────────────────────────────
        vad = get_vad()
        detected = await asyncio.to_thread(vad.detect_from_path, tmp_path)
        logger.info(f"[speech_to_text] VAD detected={detected}")

        if not detected:
            return {
                **state,
                "vad_detected":          False,
                "user_query":            "",
                "transcript_confidence": 0.0,
                "error_message":         "No speech detected in audio.",
            }

        # ── STT (reuses the same temp file) ───────────────────────────────────
        stt = get_stt()
        transcript, confidence = await asyncio.to_thread(
            stt.transcribe_from_path, tmp_path
        )
        logger.info(
            f"[speech_to_text] transcript='{transcript[:80]}…' confidence={confidence:.2f}"
        )
    finally:
        os.unlink(tmp_path)

    return {
        **state,
        "vad_detected":          True,
        "user_query":            transcript,
        "transcript_confidence": confidence,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. check_input_vulnerability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def check_input_vulnerability(state: State) -> State:
    """
    Run Llama Guard 3 on the user query.

    [PERF] Safety check (Ollama HTTP call, ~150-250 ms) and Redis memory load
    (~5 ms) have no data dependency on each other, so they run concurrently via
    asyncio.gather. The loaded memory is stored in state so that retrieve_context
    skips its Redis round-trip entirely.
    """
    query = state.get("user_query", "").strip()
    if not query:
        return {**state, "input_safety_status": "unsafe", "blocked_reason": "empty_query"}

    session_id = state.get("session_id", "")
    safety = get_safety()
    mem    = get_memory()

    # Fire safety check and memory prefetch at the same time.
    (is_safe, category), (history, summary, entities) = await asyncio.gather(
        safety.check(query, role="User"),
        asyncio.to_thread(mem.get_memory, session_id),
    )

    status: str = "safe" if is_safe else "unsafe"
    logger.info(f"[check_input_vulnerability] status={status} category={category}")

    return {
        **state,
        "input_safety_status": status,
        "blocked_reason":      category if not is_safe else "",
        # Pre-loaded memory — retrieve_context will use these directly.
        "conversation_history": history,
        "summarized_memory":    summary,
        "extracted_entities":   entities,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. handle_input_vulnerability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def handle_input_vulnerability(state: State) -> State:
    """Replace the response with a safe fallback; skip the LLM pipeline."""
    reason = state.get("blocked_reason", "policy_violation")
    logger.warning(f"[handle_input_vulnerability] Blocked input. Reason={reason}")

    return {
        **state,
        "llm_response":        cfg.SAFE_FALLBACK_MESSAGE,
        "output_safety_status": "safe",
        "reranked_context":    [],
        "retrieved_context":   [],
        "sources":             [],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. retrieve_context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def retrieve_context(state: State) -> State:
    """
    Steps:
      a) Use memory pre-loaded by check_input_vulnerability (no extra Redis call).
      b) Embed the raw query for semantic cache lookup.
      c) Check semantic cache — on a hit, skip steps d-f entirely.
      d) Reformulate the query with the small LLM.
      e) Re-embed only if the rewritten query differs from the raw query.
      f) Search Qdrant for relevant passages.
      g) Re-rank with BGE-Reranker-v2-m3.

    [PERF] Memory is already in state from check_input_vulnerability — no Redis call.
    [PERF] Semantic cache check happens before any GPU or LLM work; a hit avoids
           reformulation (~200 ms), embedding (~30 ms), Qdrant (~50 ms), reranking
           (~150 ms), and the core LLM call (~500-1000 ms).
    [PERF] The raw-query embedding computed for cache lookup is reused as the
           search embedding when the query is not reformulated, saving a second
           GPU forward pass.
    """
    session_id = state["session_id"]
    query      = state.get("user_query", "")

    # ── a) Memory (pre-loaded; fall back to Redis if somehow absent) ───────────
    history  = state.get("conversation_history") or []
    summary  = state.get("summarized_memory",  "")
    entities = state.get("extracted_entities", [])
    if not history and not summary:
        # Fallback — should not normally be needed.
        logger.warning("[retrieve_context] Memory not prefetched; loading from Redis.")
        history, summary, entities = get_memory().get_memory(session_id)

    # ── b) Embed raw query (used for cache lookup; may be reused for search) ───
    embedder  = get_embedder()
    raw_vec   = await asyncio.to_thread(embedder.embed_query, query)

    # ── c) Semantic cache check ────────────────────────────────────────────────
    cache     = get_semantic_cache()
    cached_response = await asyncio.to_thread(cache.get_by_vec, raw_vec)
    if cached_response:
        logger.info("[retrieve_context] Semantic cache HIT — skipping LLM pipeline.")
        return {
            **state,
            "llm_response":         cached_response,
            "is_cache_hit":         True,
            "rewritten_query":      query,
            "retrieved_context":    [],
            "reranked_context":     [],
            "sources":              [],
            "conversation_history": history,
            "summarized_memory":    summary,
            "extracted_entities":   entities,
            "query_embedding":      raw_vec.tolist(),
        }

    # ── d) Query reformulation ─────────────────────────────────────────────────
    llm      = get_llm()
    rewritten = await llm.reformulate_query(query, history, entities)
    logger.info(f"[retrieve_context] rewritten_query='{rewritten}'")

    # ── e) Embed — reuse raw_vec if reformulation returned the same string ─────
    if rewritten == query:
        query_vec = raw_vec   # [PERF] avoid a second GPU forward pass
    else:
        query_vec = await asyncio.to_thread(embedder.embed_query, rewritten)

    # ── f) Qdrant retrieval ────────────────────────────────────────────────────
    try:
        vs = get_vector_store()
        passages, sources = await asyncio.to_thread(
            vs.search, query_vec, cfg.TOP_K_RETRIEVE
        )
    except Exception as exc:
        logger.error(f"[retrieve_context] Qdrant search failed: {exc}")
        passages, sources = [], []

    # ── g) Re-rank ─────────────────────────────────────────────────────────────
    reranked: List[str] = []
    if passages:
        try:
            reranker = get_reranker()
            reranked, _ = await asyncio.to_thread(
                reranker.rerank, rewritten, passages, cfg.TOP_K_RERANK
            )
        except Exception as exc:
            logger.warning(f"[retrieve_context] Reranker failed: {exc}")
            reranked = passages[: cfg.TOP_K_RERANK]

    return {
        **state,
        "conversation_history": history,
        "summarized_memory":    summary,
        "extracted_entities":   entities,
        "rewritten_query":      rewritten,
        "retrieved_context":    passages,
        "reranked_context":     reranked,
        "sources":              sources,
        "is_cache_hit":         False,
        "query_embedding":      query_vec.tolist(),  # passed to save_context for cache storage
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. create_response
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def create_response(state: State) -> State:
    """
    Build a prompt from system + summary + context + history + query,
    then call the Core LLM via Google GenAI.
    """
    query       = state.get("user_query", "")
    reranked    = state.get("reranked_context", [])
    history     = state.get("conversation_history", [])
    summary     = state.get("summarized_memory", "")
    regen_count = state.get("regenerated_count", 0)

    ctx_parts: List[str] = []
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

    llm = get_llm()
    try:
        response = await llm.chat(messages)
    except Exception as exc:
        logger.error(f"[create_response] LLM call failed: {exc}")
        response = "Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời. Vui lòng thử lại."

    logger.info(f"[create_response] regen={regen_count} response_len={len(response)}")
    return {**state, "llm_response": response}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. check_output_vulnerability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def check_output_vulnerability(state: State) -> State:
    """
    Run Llama Guard 3 on the LLM's response.

    [PERF] For voice output, TTS synthesis runs in parallel with the safety
    check via asyncio.gather. Since >99% of tutor responses pass safety, the
    audio is ready immediately when the safety verdict arrives, and the
    text_to_speech node becomes a no-op. For the rare unsafe response the
    pre-generated audio bytes are simply discarded.
    """
    response      = state.get("llm_response", "")
    output_format = state.get("output_format", "text_and_voice")
    safety        = get_safety()

    if output_format == "text_and_voice" and response:
        # Run Llama Guard and TTS at the same time.
        tts = get_tts()
        (is_safe, category), audio_bytes = await asyncio.gather(
            safety.check(response, role="Agent"),
            tts.synthesize(response),
        )
        if not is_safe:
            logger.warning(
                "[check_output_vulnerability] Unsafe output; discarding pre-generated audio."
            )
            audio_bytes = b""
    else:
        is_safe, category = await safety.check(response, role="Agent")
        audio_bytes = b""

    status: str = "safe" if is_safe else "unsafe"
    logger.info(f"[check_output_vulnerability] status={status} category={category}")

    return {
        **state,
        "output_safety_status": status,
        "blocked_reason": category if not is_safe else state.get("blocked_reason", ""),
        "audio_response": audio_bytes,  # b"" on text-only or unsafe; full WAV otherwise
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. handle_output_vulnerability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def handle_output_vulnerability(state: State) -> State:
    """
    Increment regenerated_count.
    If below MAX_REGENERATIONS, the conditional edge loops back to create_response.
    If the limit is reached the edge routes forward with the fallback.
    """
    count = state.get("regenerated_count", 0) + 1
    logger.warning(
        f"[handle_output_vulnerability] Unsafe output. regen attempt "
        f"{count}/{cfg.MAX_REGENERATIONS}"
    )

    updates: State = {"regenerated_count": count}

    if count >= cfg.MAX_REGENERATIONS:
        logger.error(
            "[handle_output_vulnerability] Max regenerations reached; using fallback."
        )
        updates["llm_response"]        = cfg.SAFE_FALLBACK_MESSAGE
        updates["output_safety_status"] = "safe"

    return {**state, **updates}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. text_to_speech
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def text_to_speech(state: State) -> State:
    """
    Convert llm_response to audio with Kokoro or Zalo TTS.

    [PERF] check_output_vulnerability already synthesised the audio in parallel
    with the safety check for the happy path (safe response, voice output).
    In that case audio_response is already populated and we return immediately.
    This node only does real work for:
      - text-only output (audio_response stays b"")
      - regenerated responses after a safety failure (audio_response was cleared)
    """
    if state.get("audio_response"):
        logger.info("[text_to_speech] Audio already generated in parallel — skipping synthesis.")
        return state

    text = state.get("llm_response", "")
    if not text:
        return {**state, "audio_response": b""}

    tts = get_tts()
    try:
        audio_bytes = await tts.synthesize(text)
        logger.info(f"[text_to_speech] Generated {len(audio_bytes)} bytes of audio.")
    except Exception as exc:
        logger.error(f"[text_to_speech] TTS failed: {exc}")
        audio_bytes = b""

    return {**state, "audio_response": audio_bytes}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. save_context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def save_context(state: State) -> State:
    """
    a) Persist this turn (user question + assistant answer) to Redis memory.
    b) Store the query vector in the semantic cache so future similar questions
       get instant responses.  Only stored on genuine (non-cached) generations
       so we don't re-store stale cached responses.

    This node is called as a FastAPI background task in server.py, so neither
    Redis write nor cache store adds to the user-facing latency.
    """
    session_id = state.get("session_id", "")
    user_query = state.get("user_query", "")
    response   = state.get("llm_response", "")
    rewritten  = state.get("rewritten_query", user_query)

    # ── a) Redis memory update ─────────────────────────────────────────────────
    if user_query and response:
        mem = get_memory()
        await mem.add_turn(session_id, user_query, response)

    # ── b) Semantic cache storage (only for freshly generated responses) ───────
    if not state.get("is_cache_hit") and user_query and response:
        raw_vec = state.get("query_embedding")
        if raw_vec is not None:
            vec = np.array(raw_vec, dtype=np.float32)
            cache = get_semantic_cache()
            # Synchronous Redis write; fine inside a background task.
            cache.put_with_vec(rewritten, response, vec)
            logger.debug(f"[save_context] Cached response for query='{rewritten[:60]}'")

    logger.info(f"[save_context] Turn saved for session={session_id}")
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Conditional-edge routing functions (used in graph.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def route_after_input(state: State) -> str:
    """Skip STT when input is already text."""
    if state.get("input_format") == "voice":
        return "speech_to_text"
    return "check_input_vulnerability"


def route_after_input_safety(state: State) -> str:
    """Block unsafe inputs before they reach retrieval."""
    if state.get("input_safety_status") == "unsafe":
        return "handle_input_vulnerability"
    return "retrieve_context"


def route_after_retrieve(state: State) -> str:
    """
    [NEW] On a semantic cache hit, skip create_response and jump straight to
    check_output_vulnerability (which will run TTS in parallel with the safety
    check, just as in the normal path).
    """
    if state.get("is_cache_hit"):
        return "check_output_vulnerability"
    return "create_response"


def route_after_output_safety(state: State) -> str:
    if state.get("output_safety_status") == "unsafe":
        return "handle_output_vulnerability"
    return "text_to_speech"


def route_after_handle_output(state: State) -> str:
    """After incrementing regen counter: loop or fall through."""
    if state.get("regenerated_count", 0) < cfg.MAX_REGENERATIONS:
        return "create_response"
    return "text_to_speech"
