from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import struct
import tempfile
import time
from functools import lru_cache
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from config import cfg

logger = logging.getLogger(__name__)
load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Voice Activity Detection – Silero VAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VADService:
    """Wraps Silero VAD loaded from torch.hub."""

    def __init__(self) -> None:
        import torch
        self._torch = torch
        logger.info("Loading Silero VAD from torch.hub …")
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        (
            self.get_speech_timestamps,
            _,
            self.read_audio,
            _,
            _,
        ) = self.utils
        logger.info("Silero VAD ready.")

    # [PERF] New path-based method – accepts an already-written temp file so that
    # speech_to_text can share one temp file between VAD and STT.
    def detect_from_path(self, wav_path: str) -> bool:
        """Return True when speech is detected in *wav_path* (16 kHz PCM WAV)."""
        wav = self.read_audio(wav_path, sampling_rate=cfg.VAD_SAMPLE_RATE)
        timestamps = self.get_speech_timestamps(
            wav,
            self.model,
            threshold=cfg.VAD_THRESHOLD,
            sampling_rate=cfg.VAD_SAMPLE_RATE,
            min_speech_duration_ms=cfg.VAD_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        )
        return len(timestamps) > 0

    def detect(self, audio_bytes: bytes) -> bool:
        """Legacy bytes-based entry point (kept for backward compatibility)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            return self.detect_from_path(tmp_path)
        finally:
            os.unlink(tmp_path)


@lru_cache(maxsize=1)
def get_vad() -> VADService:
    return VADService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Speech-to-Text – faster-whisper (self-hosted)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class STTService:
    """Wraps faster-whisper for local, GPU-optional transcription."""

    def __init__(self) -> None:
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper model '{cfg.WHISPER_MODEL}' on {cfg.WHISPER_DEVICE} …")
        self.model = WhisperModel(
            cfg.WHISPER_MODEL,
            device=cfg.WHISPER_DEVICE,
            compute_type="float16" if cfg.WHISPER_DEVICE == "cuda" else "int8",
        )
        logger.info("Faster-whisper ready.")

    # [PERF] Path-based method – reuses the temp file written by speech_to_text,
    # so we avoid writing the same audio bytes to disk a second time.
    def transcribe_from_path(self, wav_path: str) -> Tuple[str, float]:
        """Return (transcript, confidence) for an already-written WAV file."""
        segments, info = self.model.transcribe(wav_path, language=cfg.WHISPER_LANGUAGE)
        text = " ".join(segment.text for segment in segments).strip()
        confidence = getattr(info, "language_probability", 0.8)
        return text, confidence  # [FIX] original had no return statement here

    def transcribe(self, audio_bytes: bytes) -> Tuple[str, float]:
        """Legacy bytes-based entry point (kept for backward compatibility).

        [FIX] The original implementation computed `text` and `confidence`
        but was missing the `return` statement, so it always returned None and
        caused a TypeError when callers tried to unpack the result.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            return self.transcribe_from_path(tmp_path)  # now correctly returns
        finally:
            os.unlink(tmp_path)


@lru_cache(maxsize=1)
def get_stt() -> STTService:
    return STTService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Text-to-Speech – Kokoro / Zalo TTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TTSService:
    """Unified TTS interface that dispatches to Kokoro or Zalo."""

    def __init__(self) -> None:
        self.backend = cfg.TTS_BACKEND
        if self.backend == "kokoro":
            from kokoro import KPipeline
            logger.info("Loading Kokoro TTS …")
            self.pipeline = KPipeline(lang_code=cfg.KOKORO_LANG_CODE)
            logger.info("Kokoro TTS ready.")
        else:
            self.pipeline = None
            logger.info(f"TTS backend = {self.backend} (API-based, no local model).")

    async def synthesize(self, text: str) -> bytes:
        """Return raw WAV bytes for *text*."""
        if self.backend == "kokoro":
            return await asyncio.to_thread(self._kokoro, text)
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

    # Thin alias used by the streaming pipeline for clarity.
    async def synthesize_sentence(self, sentence: str) -> bytes:
        """Synthesize a single sentence; identical to synthesize() but named for intent."""
        return await self.synthesize(sentence)

    def _kokoro(self, text: str) -> bytes:
        import soundfile as sf
        audio_chunks = []
        for _, _, audio_np in self.pipeline(
            text,
            voice=cfg.KOKORO_VOICE,
            speed=cfg.KOKORO_SPEED,
            split_pattern=r"\n+",
        ):
            audio_chunks.append(audio_np)
        if not audio_chunks:
            return b""
        audio = np.concatenate(audio_chunks)
        buf = io.BytesIO()
        sf.write(buf, audio, cfg.TTS_SAMPLE_RATE, format="WAV")
        return buf.getvalue()


@lru_cache(maxsize=1)
def get_tts() -> TTSService:
    return TTSService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Embeddings – BGE-M3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmbedderService:
    """BGE-M3 dense + sparse embeddings via FlagEmbedding."""

    def __init__(self) -> None:
        from FlagEmbedding import BGEM3FlagModel
        logger.info(f"Loading {cfg.EMBEDDING_MODEL} …")
        self.model = BGEM3FlagModel(
            cfg.EMBEDDING_MODEL,
            use_fp16=(cfg.EMBEDDING_DEVICE == "cuda"),
            device=cfg.EMBEDDING_DEVICE,
        )
        logger.info("BGE-M3 embedder ready.")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return dense embedding matrix (N × 1024)."""
        out = self.model.encode(
            texts,
            batch_size=cfg.EMBEDDING_BATCH_SIZE,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.array(out["dense_vecs"], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> EmbedderService:
    return EmbedderService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Re-Ranker – BGE-Reranker-v2-m3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RerankerService:
    """Cross-encoder reranker; returns top-k passages sorted by score (desc)."""

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading {cfg.RERANKER_MODEL} …")
        self.model = CrossEncoder(
            cfg.RERANKER_MODEL,
            device=cfg.RERANKER_DEVICE,
            max_length=512,
        )
        logger.info("BGE-Reranker ready.")

    def rerank(
        self, query: str, passages: List[str], top_k: int
    ) -> Tuple[List[str], List[float]]:
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs).tolist()
        ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]
        logger.info(f"Reranked top {top_k} passages with scores: {[s for _, s in top]}")
        return [p for p, _ in top], [s for _, s in top]


@lru_cache(maxsize=1)
def get_reranker() -> RerankerService:
    return RerankerService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Safety – Llama Guard 3-1B via Ollama
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SafetyService:
    """Calls Llama Guard 3 through the local Ollama API."""

    _TEMPLATE = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Task: Check if there is unsafe content in the {role} message in the "
        "conversation according to our safety policy.\n\n"
        "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
        "S1: Violence and Hate.\nS2: Sexual Content.\nS3: Criminal Planning.\n"
        "S4: Weapons.\nS5: Self-Harm.\nS6: Regulated or Controlled Substances.\n"
        "S7: Suicide & Self-Harm.\nS8: Graphic Content.\n"
        "<END UNSAFE CONTENT CATEGORIES>\n\n"
        "<BEGIN CONVERSATION>\nUser: {content}\n<END CONVERSATION>\n\n"
        "Provide your safety assessment for the {role} message. First line must read "
        "'safe' or 'unsafe'. If unsafe, on the next line start with 'S' followed by the "
        "violated category number.\n\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

    async def check(self, text: str, role: str = "User") -> Tuple[bool, str]:
        prompt = self._TEMPLATE.format(role=role, content=text)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{cfg.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": cfg.LLAMA_GUARD_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0, "num_predict": 20},
                    },
                )
            resp.raise_for_status()
            output = resp.json().get("response", "safe").strip().lower()
        except Exception as exc:
            logger.warning(f"Safety check failed ({exc}); defaulting to safe.")
            return True, ""

        first_line = output.splitlines()[0].strip()
        is_safe = first_line.startswith("safe")
        category = ""
        if not is_safe and len(output.splitlines()) > 1:
            category = output.splitlines()[1].strip()
        return is_safe, category


@lru_cache(maxsize=1)
def get_safety() -> SafetyService:
    return SafetyService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. LLM – Google GenAI via LangChain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# [PERF] Cache ChatGoogleGenerativeAI instances keyed on (model, temperature,
# max_tokens). The original code constructed a new object on EVERY .chat() call,
# which re-initialises the underlying HTTP transport each time.
@lru_cache(maxsize=8)
def _get_google_client(
    model: str, temperature: float, max_tokens: int
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class LLMService:

    async def chat(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            # [PERF] Reuse cached client; no longer re-instantiated per call.
            llm = _get_google_client(
                model=model or cfg.CORE_LLM_MODEL,
                temperature=temperature if temperature is not None else cfg.LLM_TEMPERATURE,
                max_tokens=max_tokens if max_tokens is not None else cfg.LLM_MAX_TOKENS,
            )
            response = await llm.ainvoke(messages)

            if isinstance(response.content, list):
                return " ".join(
                    part.get("text", "") for part in response.content
                ).strip()
            return response.content.strip()
        except Exception as exc:
            logger.error(f"LLM chat failed: {exc}.")
            raise

    # ── [NEW] Streaming variant ───────────────────────────────────────────────
    async def stream_chat(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields text tokens from the LLM as they arrive.

        Uses LangChain's .astream() on the cached ChatGoogleGenerativeAI client,
        so the HTTP connection is reused and there is no per-call re-initialisation.

        Typical usage:
            async for token in llm.stream_chat(messages):
                buffer += token
        """
        llm = _get_google_client(
            model=model or cfg.CORE_LLM_MODEL,
            temperature=temperature if temperature is not None else cfg.LLM_TEMPERATURE,
            max_tokens=max_tokens if max_tokens is not None else cfg.LLM_MAX_TOKENS,
        )
        try:
            async for chunk in llm.astream(messages):
                content = chunk.content
                if not content:
                    continue
                # content can be a string or a list of typed parts (multimodal)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text", "")
                        else:
                            text = str(part)
                        if text:
                            yield text
                else:
                    yield content
        except Exception as exc:
            logger.error(f"[LLMService.stream_chat] Streaming failed: {exc}")
            raise

    async def reformulate_query(
        self,
        query: str,
        history: List[dict],
        entities: List[str],
    ) -> str:
        """Use a smaller local model to rewrite the query for better retrieval."""
        clean_history = [
            m for m in history if m.get("role") in ("user", "assistant")
        ]
        recent = clean_history[-4:] if clean_history else []

        entity_instruction = ""
        if entities:
            entity_instruction = f"\nImportant terms to include: {', '.join(entities[:5])}"

        query_lower = query.lower()
        is_short = len(query.split()) < 5
        has_pronouns = any(
            word in query_lower
            for word in ["it", "this", "that", "these", "those", "how", "what about", "explain"]
        )

        if not (is_short or has_pronouns):
            if entities:
                return f"{query} {' '.join(entities[:3])}"
            return query

        if recent:
            history_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:200]}"
                for m in recent
            )
            user_msg = (
                f"Previous conversation:\n{history_text}\n\n"
                f"Current question: {query}"
                f"{entity_instruction}\n\n"
                "Write a search query that works without the conversation context:"
            )
        else:
            if entities:
                return f"{query} {' '.join(entities[:3])}"
            return query

        try:
            result = await self.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Rewrite questions as standalone search queries. Output only the query.",
                    },
                    {"role": "user", "content": user_msg},
                ],
                model=cfg.REFORMULATION_MODEL,
                temperature=0.1,
                max_tokens=100,
            )
            return result if result and len(result) >= 3 else query
        except Exception:
            return query


@lru_cache(maxsize=1)
def get_llm() -> LLMService:
    return LLMService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Vector Store – Qdrant
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VectorStoreService:

    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        logger.info(f"Connecting to Qdrant at {cfg.QDRANT_PATH} …")
        self.client = QdrantClient(path=cfg.QDRANT_PATH, timeout=30)
        logger.info("Qdrant client ready.")

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[List[str], List[str]]:
        results = self.client.query_points(
            collection_name=cfg.QDRANT_COLLECTION,
            query=query_vec.tolist(),
            limit=top_k,
            with_payload=True,
            score_threshold=0.7,
        )
        passages = [r.payload.get("text", "") for r in results.points]
        sources = [r.payload.get("source", "unknown") for r in results.points]
        return passages, sources


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Semantic Cache  [NEW]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SemanticCacheService:
    """
    Redis-backed semantic similarity cache.

    On a cache hit, the pipeline skips query reformulation, Qdrant search,
    reranking, and the core LLM call entirely — saving 800-1500 ms per turn.

    Storage format: one Redis key holds a JSON list of
        {"query": str, "response": str, "vec": List[float]}
    sorted oldest → newest, capped at MAX_ENTRIES.

    Lookup is O(N) cosine similarity in NumPy. For a tutor workload with
    MAX_ENTRIES ≤ 500 this takes < 2 ms (BGE-M3 vecs are L2-normalised so
    the dot product IS the cosine similarity).

    For larger deployments, replace with a Qdrant collection or RediSearch
    vector index.
    """

    KEY = "semantic_cache:v1"
    MAX_ENTRIES = 500
    SIMILARITY_THRESHOLD = 0.92   # conservative — only serve very close matches
    TTL = 60 * 60 * 24 * 7        # 7 days

    def __init__(self) -> None:
        import redis
        self._redis = redis.from_url(cfg.REDIS_URL, decode_responses=True)

    # ── lookup ────────────────────────────────────────────────────────────────

    def get_by_vec(self, query_vec: np.ndarray) -> Optional[str]:
        """
        Return the cached response whose stored query vector is closest to
        *query_vec*, provided the cosine similarity exceeds SIMILARITY_THRESHOLD.
        Returns None on a miss.
        """
        raw = self._redis.get(self.KEY)
        if not raw:
            return None
        entries = json.loads(raw)
        if not entries:
            return None

        vecs = np.array([e["vec"] for e in entries], dtype=np.float32)  # (N, dim)
        # BGE-M3 outputs L2-normalised vectors → dot product == cosine similarity
        sims = vecs @ query_vec  # (N,)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.SIMILARITY_THRESHOLD:
            logger.info(
                f"[SemanticCache] HIT  similarity={best_sim:.4f}  "
                f"query='{entries[best_idx]['query'][:60]}'"
            )
            return entries[best_idx]["response"]

        logger.debug(f"[SemanticCache] MISS best_sim={best_sim:.4f}")
        return None

    # ── store ─────────────────────────────────────────────────────────────────

    def put_with_vec(
        self, query: str, response: str, vec: np.ndarray
    ) -> None:
        """Persist a (query, response, vector) triple in the cache."""
        raw = self._redis.get(self.KEY)
        entries: list = json.loads(raw) if raw else []

        entries.append(
            {"query": query, "response": response, "vec": vec.tolist()}
        )

        # Keep only the most recent MAX_ENTRIES entries
        if len(entries) > self.MAX_ENTRIES:
            entries = entries[-self.MAX_ENTRIES:]

        self._redis.set(self.KEY, json.dumps(entries), ex=self.TTL)
        logger.debug(f"[SemanticCache] Stored entry (total={len(entries)})")


@lru_cache(maxsize=1)
def get_semantic_cache() -> SemanticCacheService:
    return SemanticCacheService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Conversation Memory – Local Redis-based
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MemoryService:
    """
    Fully local memory using Redis.
      - conversation_history : last N turns stored as JSON in Redis
      - summarized_memory    : LLM-generated rolling summary (auto-triggered)
      - extracted_entities   : LLM-extracted entities from the last turn
    """

    HISTORY_KEY = "mem:history:{sid}"
    SUMMARY_KEY = "mem:summary:{sid}"
    ENTITY_KEY  = "mem:entities:{sid}"
    MAX_TURNS   = 10
    SUMMARY_TTL = 60 * 60 * 24 * 7   # 7 days

    def __init__(self) -> None:
        import redis
        self._redis = redis.from_url(cfg.REDIS_URL, decode_responses=True)
        self._json  = json

    def _hkey(self, sid: str) -> str: return self.HISTORY_KEY.format(sid=sid)
    def _skey(self, sid: str) -> str: return self.SUMMARY_KEY.format(sid=sid)
    def _ekey(self, sid: str) -> str: return self.ENTITY_KEY.format(sid=sid)

    async def _llm_summarize(self, history: list, old_summary: str) -> str:
        llm = get_llm()
        old = f"Previous summary:\n{old_summary}\n\n" if old_summary else ""
        turns = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history
        )
        prompt = (
            f"{old}New conversation turns:\n{turns}\n\n"
            "Write a concise updated summary (≤120 words) capturing key facts, "
            "topics discussed, and any decisions made. Output ONLY the summary."
        )
        return await llm.chat(
            [{"role": "user", "content": prompt}],
            model=cfg.REFORMULATION_MODEL,
            temperature=0.0,
            max_tokens=200,
        )

    async def _llm_extract_entities(
        self, user_msg: str, assistant_msg: str
    ) -> list[str]:
        llm = get_llm()
        prompt = (
            "Extract named entities (people, subjects, concepts, places) from:\n"
            f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
            "Return a comma-separated list only. Example: Newton, gravity, 1687"
        )
        raw = await llm.chat(
            [{"role": "user", "content": prompt}],
            model=cfg.REFORMULATION_MODEL,
            temperature=0.0,
            max_tokens=80,
        )
        return [e.strip() for e in raw.split(",") if e.strip()]

    # ── public API ─────────────────────────────────────────────────────────────

    def get_memory(self, session_id: str):
        """
        Load history, summary, and entities from Redis.

        [PERF] Uses a Redis pipeline so all three GETs are sent in a single
        round-trip instead of three sequential ones.
        """
        pipe = self._redis.pipeline()
        pipe.get(self._hkey(session_id))
        pipe.get(self._skey(session_id))
        pipe.get(self._ekey(session_id))
        raw_hist, raw_sum, raw_ents = pipe.execute()

        history  = self._json.loads(raw_hist) if raw_hist else []
        summary  = raw_sum if raw_sum else ""
        entities = self._json.loads(raw_ents) if raw_ents else []
        return history, summary, entities

    async def add_turn(
        self, session_id: str, user_msg: str, assistant_msg: str
    ) -> None:
        history, summary, _ = self.get_memory(session_id)

        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant",  "content": assistant_msg})

        if len(history) > self.MAX_TURNS * 2:
            # [PERF] Summarise and extract entities in parallel (both are LLM
            # calls with no data dependency on each other).
            summary, entities = await asyncio.gather(
                self._llm_summarize(history, summary),
                self._llm_extract_entities(user_msg, assistant_msg),
            )
            history = history[-(self.MAX_TURNS * 2):]
        else:
            # No summarisation needed; entity extraction only.
            entities = await self._llm_extract_entities(user_msg, assistant_msg)

        # [PERF] Write all three keys in a single Redis pipeline.
        pipe = self._redis.pipeline()
        pipe.set(self._hkey(session_id), self._json.dumps(history),   ex=self.SUMMARY_TTL)
        pipe.set(self._skey(session_id), summary,                      ex=self.SUMMARY_TTL)
        pipe.set(self._ekey(session_id), self._json.dumps(entities),  ex=self.SUMMARY_TTL)
        pipe.execute()


@lru_cache(maxsize=1)
def get_memory() -> MemoryService:
    return MemoryService()