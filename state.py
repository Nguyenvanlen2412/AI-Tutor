"""
state.py – typed state shared across all LangGraph nodes.
"""

from typing import TypedDict, Optional, List, Dict, Literal, Any


class State(TypedDict, total=False):
    # ── INPUT ──────────────────────────────────────────────────────────────────
    input_format: Literal["text", "voice"]
    raw_audio: Optional[bytes]       # PCM / WAV bytes when input_format=="voice"
    user_query: str                  # final text query (after STT if applicable)

    # ── AUDIO PIPELINE ─────────────────────────────────────────────────────────
    vad_detected: bool               # True if Silero VAD found speech
    transcript_confidence: float     # Whisper avg log-prob → confidence 0-1

    # ── MEMORY ─────────────────────────────────────────────────────────────────
    conversation_history: List[Dict[str, str]]   # [{"role": "user"|"assistant", "content": "..."}]

    # ── SAFETY ─────────────────────────────────────────────────────────────────
    input_safety_status: Literal["pending", "safe", "unsafe"]
    output_safety_status: Literal["pending", "safe", "unsafe"]
    blocked_reason: str              # Llama Guard category string if flagged

    # ── QUERY PROCESSING ───────────────────────────────────────────────────────
    rewritten_query: str             # reformulated query from Gemma 3B

    # ── RETRIEVAL ──────────────────────────────────────────────────────────────
    is_cache_hit: bool               # True if Redis semantic cache returned a hit
    cached_response: str             # cached LLM response text
    retrieved_context: List[str]     # raw Qdrant passages
    reranked_context: List[str]      # BGE-Reranker top-k passages
    sources: List[str]               # document names / URLs for citations

    # ── LLM ────────────────────────────────────────────────────────────────────
    llm_response: str
    regenerated_count: int           # how many times output safety triggered regen

    # ── OUTPUT ─────────────────────────────────────────────────────────────────
    output_format: Literal["text_and_voice"]
    audio_response: Optional[bytes]  # WAV bytes when output_format=="voice"

    # ── META ───────────────────────────────────────────────────────────────────
    session_id: str
    user_id: str
    latency_ms: int
    error_message: str
