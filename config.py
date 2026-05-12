"""
config.py – centralised configuration for the AI Tutor pipeline.
Override any value via environment variables or a .env file.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── Whisper STT ────────────────────────────────────────────────────────────
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    # Options: tiny | base | small | medium | large-v3
    WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "vi")  # Vietnamese default
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")     # or "cuda"

    # ── Silero VAD ─────────────────────────────────────────────────────────────
    VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
    VAD_SAMPLE_RATE: int = int(os.getenv("VAD_SAMPLE_RATE", "16000"))
    VAD_MIN_SPEECH_DURATION_MS: int = int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", "250"))

    # ── Ollama (Core LLM + Query Reformulation + Safety) ─────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    CORE_LLM_MODEL: str = os.getenv("CORE_LLM_MODEL", "gemma3:1b-it-qat")
    REFORMULATION_MODEL: str = os.getenv("REFORMULATION_MODEL", "gemma3:270m")
    LLAMA_GUARD_MODEL: str = os.getenv("LLAMA_GUARD_MODEL", "llama-guard3:1b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # ── BGE-M3 Embeddings ──────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")  # or "cuda"
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "12"))

    # ── BGE Reranker ───────────────────────────────────────────────────────────
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANKER_DEVICE: str = os.getenv("RERANKER_DEVICE", "cpu")

    # ── Qdrant ─────────────────────────────────────────────────────────────────
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "ai_tutor_docs")
    QDRANT_VECTOR_SIZE: int = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))  # BGE-M3 dense dim

    # ── Retrieval ──────────────────────────────────────────────────────────────
    TOP_K_RETRIEVE: int = int(os.getenv("TOP_K_RETRIEVE", "10"))
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "3"))
    HYBRID_SEARCH_ALPHA: float = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.7"))  # 1=dense, 0=sparse

    # ── Redis Semantic Cache ───────────────────────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.8"))
    CACHE_KEY_PREFIX: str = "ai_tutor:cache:"

    # ── Zep Memory ────────────────────────────────────────────────────────────
    MEMORY_WINDOW: int = int(os.getenv("MEMORY_WINDOW", "10"))

    # ── TTS ───────────────────────────────────────────────────────────────────
    TTS_BACKEND: str = os.getenv("TTS_BACKEND", "kokoro")       # kokoro | zalo | viettts
    KOKORO_LANG_CODE: str = os.getenv("KOKORO_LANG_CODE", "a")  # 'a'=en, 'z'=zh, etc.
    KOKORO_VOICE: str = os.getenv("KOKORO_VOICE", "af_heart")
    KOKORO_SPEED: float = float(os.getenv("KOKORO_SPEED", "1.4"))
    ZALO_TTS_API_KEY: str = os.getenv("ZALO_TTS_API_KEY", "")
    ZALO_TTS_SPEAKER_ID: str = os.getenv("ZALO_TTS_SPEAKER_ID", "1")
    TTS_SAMPLE_RATE: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))

    # ── Safety ────────────────────────────────────────────────────────────────
    MAX_REGENERATIONS: int = int(os.getenv("MAX_REGENERATIONS", "3"))
    SAFE_FALLBACK_MESSAGE: str = (
        "Xin lỗi, tôi không thể trả lời câu hỏi đó. "
        "Hãy thử hỏi một câu hỏi khác liên quan đến học tập."
    )

    # ── Document Ingestion ────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    DOCS_DIR: str = os.getenv("DOCS_DIR", "./docs")


# Singleton instance imported everywhere
cfg = Config()
