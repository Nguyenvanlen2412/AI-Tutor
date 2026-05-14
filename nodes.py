"""
nodes.py – every LangGraph node for the AI Tutor pipeline.

Node execution order (see graph.py for edges):
  get_user_input
    ↓ (voice?) speech_to_text
  check_input_vulnerability
    ↓ (unsafe?) handle_input_vulnerability
  retrieve_context
  check_output_vulnerability
    ↓ (unsafe & retries left?) handle_output_vulnerability → create_response (loop)
  text_to_speech          ← skipped when output_format == "text"
  save_context
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import List

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
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT for the Core LLM (edit to suit your subject domain)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent, friendly, and patient AI tutor.

Your task is to help students understand concepts, solve problems, and answer questions

clearly, accurately, and in a way that encourages critical thinking.

Instructions:

- Answer in the language the student uses (English).

- Use simple, concise language and break down complex ideas into easy-to-understand steps.

- Do not add emojis or informal slang.

- Do not give links or references to external resources; instead, explain concepts directly.

- If context is provided in the material, prioritize using that information.

- Cite sources if necessary.

- Be concise by default. Answer in 5-7 sentences unless the user explicitly asks for detail.

- Avoid long introductions, disclaimers, and unnecessary framing.

- If unsure, be upfront and suggest the student consult additional resources.

- Encourage independent thinking by asking appropriate feedback questions.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. get_user_input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_user_input(state: State) -> State:
    """
    Capture the initial payload.
    Assigns session_id / user_id if not already set.
    Expected caller to pre-fill:
      state["input_format"]  – "text" | "voice"
      state["user_query"]    – the text (if text input)
      state["raw_audio"]     – raw WAV bytes (if voice input)
      state["output_format"] – "text and voice"
      state["user_id"]       – caller-supplied user identifier
    """
    updates: State = {}

    if not state.get("session_id"):
        updates["session_id"] = str(uuid.uuid4())
    if not state.get("user_id"):
        updates["user_id"] = "anonymous"
    if not state.get("input_format"):
        updates["input_format"] = "text"
    if not state.get("output_format"):
        updates["output_format"] = state.get("output_format", "text_and_voice")

    # Reset per-turn fields
    updates.update(
        {
            "input_safety_status": "pending",
            "output_safety_status": "pending",
            "regenerated_count": state.get("regenerated_count", 0),
            "error_message": "",
            "blocked_reason": "",
            "vad_detected": False,
            "transcript_confidence": 0.0,
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

def speech_to_text(state: State) -> State:
    """
    1. Run Silero VAD to confirm speech is present.
    2. If detected, transcribe with Whisper.
    3. Populate state["user_query"] and state["transcript_confidence"].
    """
    audio_bytes: bytes = state.get("raw_audio", b"")
    if not audio_bytes:
        logger.warning("[speech_to_text] No audio bytes found.")
        return {**state, "error_message": "No audio provided.", "vad_detected": False}

    # ── VAD ────────────────────────────────────────────────────────────────────
    vad = get_vad()
    detected = vad.detect(audio_bytes)
    logger.info(f"[speech_to_text] VAD detected={detected}")

    if not detected:
        return {
            **state,
            "vad_detected": False,
            "user_query": "",
            "transcript_confidence": 0.0,
            "error_message": "No speech detected in audio.",
        }

    # ── Transcription ──────────────────────────────────────────────────────────
    stt = get_stt()
    transcript, confidence = stt.transcribe(audio_bytes)
    logger.info(
        f"[speech_to_text] transcript='{transcript[:80]}…' confidence={confidence:.2f}"
    )

    return {
        **state,
        "vad_detected": True,
        "user_query": transcript,
        "transcript_confidence": confidence,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. check_input_vulnerability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_input_vulnerability(state: State) -> State:
    """Run Llama Guard 3 on the user query."""
    query = state.get("user_query", "").strip()
    if not query:
        return {**state, "input_safety_status": "unsafe", "blocked_reason": "empty_query"}

    safety = get_safety()
    is_safe, category = safety.check(query, role="User")
    status: str = "safe" if is_safe else "unsafe"
    logger.info(f"[check_input_vulnerability] status={status} category={category}")

    return {
        **state,
        "input_safety_status": status,
        "blocked_reason": category if not is_safe else "",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. handle_input_vulnerability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def handle_input_vulnerability(state: State) -> State:
    """
    Replace the response with a safe fallback.
    This node terminates the retrieval/LLM pipeline – the graph will jump
    straight to text_to_speech / save_context.
    """
    reason = state.get("blocked_reason", "policy_violation")
    logger.warning(f"[handle_input_vulnerability] Blocked input. Reason={reason}")

    return {
        **state,
        "llm_response": cfg.SAFE_FALLBACK_MESSAGE,
        "output_safety_status": "safe",   # fallback is pre-approved
        "reranked_context": [],
        "retrieved_context": [],
        "sources": [],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. retrieve_context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def retrieve_context(state: State) -> State:
    """
    Steps:
      a) Load conversation history + summary + entities from redis.
      b) Reformulate the query.
      c) Embed the rewritten query with BGE-M3.
      e) Search Qdrant for relevant passages.
      f) Re-rank with BGE-Reranker-v2-m3.
    """
    session_id = state["session_id"]
    query = state.get("user_query", "")

    # ── a) Load memory  ────────────────────────────────────────────────
    mem = get_memory()
    history, summary, entities = mem.get_memory(session_id)
    logger.info(
        f"[retrieve_context] history={len(history)} turns, "
        f"entities={entities[:5]}"
    )

    # ── b) Query reformulation ─────────────────────────────────────────────────
    llm = get_llm()
    rewritten = llm.reformulate_query(query, history, entities)
    logger.info(f"[retrieve_context] rewritten_query='{rewritten}'")

    # ── c) Embed ───────────────────────────────────────────────────────────────
    embedder = get_embedder()
    query_vec = embedder.embed_query(rewritten)

    # ── e) Qdrant retrieval ────────────────────────────────────────────────────
    try:
        vs = get_vector_store()
        passages, sources = vs.search(query_vec, top_k=cfg.TOP_K_RETRIEVE)
    except Exception as exc:
        logger.error(f"Qdrant search failed: {exc}")
        passages, sources = [], []

    # ── f) Re-rank ─────────────────────────────────────────────────────────────
    if passages:
        try:
            reranker = get_reranker()
            reranked, _ = reranker.rerank(rewritten, passages, top_k=cfg.TOP_K_RERANK)
        except Exception as exc:
            logger.warning(f"Reranker failed: {exc}")
            reranked = passages[: cfg.TOP_K_RERANK]
    else:
        reranked = []

    return {
        **state,
        "conversation_history": history,
        "summarized_memory": summary,
        "extracted_entities": entities,
        "rewritten_query": rewritten,
        "retrieved_context": passages,
        "reranked_context": reranked,
        "sources": sources,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. create_response
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_response(state: State) -> State:
    """
    Build a prompt from:
      - system prompt
      - summary (long-term memory)
      - re-ranked context passages
      - conversation history (last N turns)
      - current user query
    Then call the Core LLM via Ollama.
    If this is a regeneration pass, append an instruction to keep it safe.
    """
    query = state.get("user_query", "")
    reranked = state.get("reranked_context", [])
    sources = state.get("sources", [])
    history = state.get("conversation_history", [])
    summary = state.get("summarized_memory", "")
    regen_count = state.get("regenerated_count", 0)

    # ── Build context block ────────────────────────────────────────────────────
    ctx_parts: List[str] = []
    if summary:
        ctx_parts.append(f"[previous conversation summary]\n{summary}")
    if reranked:
        ctx_str = "\n\n".join(
            f"[Document {i+1}] {p}" for i, p in enumerate(reranked)
        )
        ctx_parts.append(f"[Context from documents]\n{ctx_str}")

    # ── System message ─────────────────────────────────────────────────────────
    system_content = SYSTEM_PROMPT
    if ctx_parts:
        system_content += "\n\n" + "\n\n".join(ctx_parts)
    if regen_count > 0:
        system_content += (
            "\n\n[IMPORTANT] The previous response violated the safety policy. "
            "Please provide a completely safe, appropriate, and helpful response."
        )

    # ── Assemble messages list ─────────────────────────────────────────────────
    messages = [{"role": "system", "content": system_content}]
    messages.extend(history[-cfg.MEMORY_WINDOW:])
    messages.append({"role": "user", "content": query})

    # ── Call LLM ───────────────────────────────────────────────────────────────
    llm = get_llm()
    try:
        response = llm.chat(messages)
    except Exception as exc:
        logger.error(f"[create_response] LLM call failed: {exc}")
        response = "Sorry, an error occurred while generating the response. Please try again."

    logger.info(f"[create_response] regen={regen_count} response_len={len(response)}")

    return {**state, "llm_response": response}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━══
# 7. check_output_vulnerability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_output_vulnerability(state: State) -> State:
    """Run Llama Guard 3 on the LLM's response."""
    response = state.get("llm_response", "")
    safety = get_safety()
    is_safe, category = safety.check(response, role="Agent")
    status: str = "safe" if is_safe else "unsafe"
    logger.info(f"[check_output_vulnerability] status={status} category={category}")

    return {
        **state,
        "output_safety_status": status,
        "blocked_reason": category if not is_safe else state.get("blocked_reason", ""),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. handle_output_vulnerability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def handle_output_vulnerability(state: State) -> State:
    """
    Increment regenerated_count.
    If below MAX_REGENERATIONS, the conditional edge will loop back to
    create_response with the regen instruction already embedded.
    If the limit is reached the edge routes forward with the fallback.
    """
    count = state.get("regenerated_count", 0) + 1
    logger.warning(
        f"[handle_output_vulnerability] Unsafe output. regen attempt {count}/{cfg.MAX_REGENERATIONS}"
    )

    updates: State = {"regenerated_count": count}

    if count >= cfg.MAX_REGENERATIONS:
        logger.error("[handle_output_vulnerability] Max regenerations reached; using fallback.")
        updates["llm_response"] = cfg.SAFE_FALLBACK_MESSAGE
        updates["output_safety_status"] = "safe"

    return {**state, **updates}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. text_to_speech  (only reached when output_format == "voice")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def text_to_speech(state: State) -> State:
    """Convert llm_response to audio with Kokoro or Zalo TTS."""
    text = state.get("llm_response", "")
    if not text:
        return {**state, "audio_response": b""}  # Return empty bytes, not None

    tts = get_tts()
    try:
        audio_bytes = tts.synthesize(text)
        logger.info(f"[text_to_speech] Generated {len(audio_bytes)} bytes of audio.")
    except Exception as exc:
        logger.error(f"[text_to_speech] TTS failed: {exc}")
        audio_bytes = b""  # Return empty bytes instead of None

    return {**state, "audio_response": audio_bytes}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. save_context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_context(state: State) -> State:
    """
    a) Persist this turn (user question + assistant answer).
    c) Record end-to-end latency in state.
    """
    session_id = state.get("session_id", "")
    user_query = state.get("user_query", "")
    response = state.get("llm_response", "")
    rewritten = state.get("rewritten_query", user_query)

    # ── a) ─────────────────────────────────────────────────────────────────
    if user_query and response:
        mem = get_memory()
        mem.add_turn(session_id, user_query, response)

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


def route_after_output_safety(state: State) -> str:
    """
    Loop back for regeneration, or proceed.
    Output path: unsafe & retries remain → handle_output_vulnerability
                 safe (or retries exhausted) + voice → text_to_speech
                 safe (or retries exhausted) + text  → save_context
    """
    if state.get("output_safety_status") == "unsafe":
        return "handle_output_vulnerability"
    return "text_to_speech"


def route_after_handle_output(state: State) -> str:
    """After incrementing regen counter: loop or fall through."""
    if state.get("regenerated_count", 0) < cfg.MAX_REGENERATIONS:
        return "create_response"
    # Limit reached – fallback already written, skip to TTS / save
    return "text_to_speech"
