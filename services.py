"""
services.py – lazy-loaded singleton wrappers around every external service.

All heavy models are loaded once on first use to avoid slowing startup.
"""

from __future__ import annotations

import io
import json
import logging
import struct
import time
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import requests

from config import cfg

logger = logging.getLogger(__name__)


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

    def detect(self, audio_bytes: bytes) -> bool:
        """Return True when speech is detected in *audio_bytes* (16 kHz PCM WAV)."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            wav = self.read_audio(tmp_path, sampling_rate=cfg.VAD_SAMPLE_RATE)
            timestamps = self.get_speech_timestamps(
                wav,
                self.model,
                threshold=cfg.VAD_THRESHOLD,
                sampling_rate=cfg.VAD_SAMPLE_RATE,
                min_speech_duration_ms=cfg.VAD_MIN_SPEECH_DURATION_MS,
            )
            return len(timestamps) > 0
        finally:
            os.unlink(tmp_path)


@lru_cache(maxsize=1)
def get_vad() -> VADService:
    return VADService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Speech-to-Text – OpenAI Whisper (self-hosted)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class STTService:
    """Wraps openai-whisper for local, GPU-optional transcription."""

    def __init__(self) -> None:
        import whisper
        logger.info(f"Loading Whisper model '{cfg.WHISPER_MODEL}' on {cfg.WHISPER_DEVICE} …")
        self.model = whisper.load_model(cfg.WHISPER_MODEL, device=cfg.WHISPER_DEVICE)
        logger.info("Whisper ready.")

    def transcribe(self, audio_bytes: bytes) -> Tuple[str, float]:
        """
        Returns (transcript, confidence).
        confidence is the mean of segment-level avg_logprobs mapped to [0, 1].
        """
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            result = self.model.transcribe(
                tmp_path,
                language=cfg.WHISPER_LANGUAGE,
                fp16=(cfg.WHISPER_DEVICE == "cuda"),
            )
            text = result["text"].strip()
            # avg_logprob is ≤ 0; map to roughly 0-1
            segs = result.get("segments", [])
            if segs:
                avg_lp = sum(s["avg_logprob"] for s in segs) / len(segs)
                confidence = float(np.clip(np.exp(avg_lp), 0.0, 1.0))
            else:
                confidence = 0.0
            return text, confidence
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

    def synthesize(self, text: str) -> bytes:
        """Return raw WAV bytes for *text*."""
        if self.backend == "kokoro":
            return self._kokoro(text)
        elif self.backend == "zalo":
            return self._zalo(text)
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

    # ── Kokoro ─────────────────────────────────────────────────────────────────
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

    # ── Zalo TTS API ───────────────────────────────────────────────────────────
    def _zalo(self, text: str) -> bytes:
        resp = requests.post(
            "https://api.zalo.ai/v1/tts/synthesize",
            headers={"apikey": cfg.ZALO_TTS_API_KEY, "Content-Type": "application/x-www-form-urlencoded"},
            data={"input": text, "speakerId": cfg.ZALO_TTS_SPEAKER_ID},
            timeout=30,
        )
        resp.raise_for_status()
        audio_url = resp.json()["data"]["url"]
        audio_resp = requests.get(audio_url, timeout=30)
        audio_resp.raise_for_status()
        return audio_resp.content


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
        from FlagEmbedding import FlagReranker
        logger.info(f"Loading {cfg.RERANKER_MODEL} …")
        self.model = FlagReranker(
            cfg.RERANKER_MODEL,
            use_fp16=(cfg.RERANKER_DEVICE == "cuda"),
            device=cfg.RERANKER_DEVICE,
        )
        logger.info("BGE-Reranker ready.")

    def rerank(self, query: str, passages: List[str], top_k: int) -> Tuple[List[str], List[float]]:
        pairs = [[query, p] for p in passages]
        scores = self.model.compute_score(pairs, normalize=True)
        ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]
        return [p for p, _ in top], [s for _, s in top]


@lru_cache(maxsize=1)
def get_reranker() -> RerankerService:
    return RerankerService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Safety – Llama Guard 3-1B via Ollama
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SafetyService:
    """
    Calls Llama Guard 3 through the local Ollama API.
    Returns (is_safe: bool, category: str).
    """

    # Llama Guard 3 prompt template
    _TEMPLATE = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Task: Check if there is unsafe content in the {role} message in the "
        "conversation according to our safety policy.\n\n"
        "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
        "S1: Violence and Hate.\n"
        "S2: Sexual Content.\n"
        "S3: Criminal Planning.\n"
        "S4: Weapons.\n"
        "S5: Self-Harm.\n"
        "S6: Regulated or Controlled Substances.\n"
        "S7: Suicide & Self-Harm.\n"
        "S8: Graphic Content.\n"
        "<END UNSAFE CONTENT CATEGORIES>\n\n"
        "<BEGIN CONVERSATION>\nUser: {content}\n<END CONVERSATION>\n\n"
        "Provide your safety assessment for the {role} message. First line must read "
        "'safe' or 'unsafe'. If unsafe, on the next line start with 'S' followed by the "
        "violated category number.\n\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

    def check(self, text: str, role: str = "User") -> Tuple[bool, str]:
        prompt = self._TEMPLATE.format(role=role, content=text)
        try:
            resp = requests.post(
                f"{cfg.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": cfg.LLAMA_GUARD_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 20},
                },
                timeout=30,
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
# 7. LLM – Ollama (Core + Query Reformulation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMService:
    """Thin wrapper around Ollama's /api/chat endpoint."""

    def chat(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            payload = {
                "model": model or cfg.CORE_LLM_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature if temperature is not None else cfg.LLM_TEMPERATURE,
                    "num_predict": max_tokens or cfg.LLM_MAX_TOKENS,
                },
            }
            resp = requests.post(
                f"{cfg.OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as exc:
            logger.error(f"LLM chat failed: {exc}. Is Ollama running at {cfg.OLLAMA_BASE_URL}?")
            raise

    def reformulate_query(
        self,
        query: str,
        history: List[dict],
        entities: List[str],
    ) -> str:
        """Use a smaller local model to rewrite the query for better retrieval."""
        entity_str = ", ".join(entities) if entities else "none"
        system = (
            "You are a search query optimizer. "
            "Rewrite the user's latest question into a concise, self-contained search query "
            "that incorporates relevant context from the conversation. "
            "Output ONLY the rewritten query, no explanation."
        )
        history_snippet = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]
        )
        user_msg = (
            f"Conversation so far:\n{history_snippet}\n\n"
            f"Known entities: {entity_str}\n\n"
            f"Latest question: {query}\n\n"
            "Rewritten query:"
        )
        return self.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            model=cfg.REFORMULATION_MODEL,
            temperature=0.0,
            max_tokens=128,
        )


@lru_cache(maxsize=1)
def get_llm() -> LLMService:
    return LLMService()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Vector Store – Qdrant
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VectorStoreService:
    """Hybrid search (dense + BM25 sparse) via Qdrant."""

    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        logger.info(f"Connecting to Qdrant at {cfg.QDRANT_URL} …")
        self.client = QdrantClient(
            url=cfg.QDRANT_URL,
            api_key=cfg.QDRANT_API_KEY or None,
            timeout=30,
        )
        logger.info("Qdrant client ready.")

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[List[str], List[str]]:
        """
        Returns (passages, sources).
        Uses dense vector similarity; extend to hybrid if SparseVectors are indexed.
        """
        from qdrant_client.models import SearchRequest

        results = self.client.search(
            collection_name=cfg.QDRANT_COLLECTION,
            query_vector=query_vec.tolist(),
            limit=top_k,
            with_payload=True,
        )
        passages = [r.payload.get("text", "") for r in results]
        sources = [r.payload.get("source", "unknown") for r in results]
        return passages, sources


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


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
    HISTORY_KEY  = "mem:history:{sid}"
    SUMMARY_KEY  = "mem:summary:{sid}"
    ENTITY_KEY   = "mem:entities:{sid}"
    MAX_TURNS    = 10    # keep in Redis before summarising
    SUMMARY_TTL  = 60 * 60 * 24 * 7   # 7 days

    def __init__(self) -> None:
        import redis, json
        self._redis = redis.from_url(cfg.REDIS_URL, decode_responses=True)
        self._json  = json

    # ── helpers ────────────────────────────────────────────────────────────────
    def _hkey(self, sid): return self.HISTORY_KEY.format(sid=sid)
    def _skey(self, sid): return self.SUMMARY_KEY.format(sid=sid)
    def _ekey(self, sid): return self.ENTITY_KEY.format(sid=sid)

    def _llm_summarize(self, history: list, old_summary: str) -> str:
        llm = get_llm()
        old = f"Previous summary:\n{old_summary}\n\n" if old_summary else ""
        turns = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
        prompt = (
            f"{old}New conversation turns:\n{turns}\n\n"
            "Write a concise updated summary (≤120 words) capturing key facts, "
            "topics discussed, and any decisions made. Output ONLY the summary."
        )
        return llm.chat(
            [{"role": "user", "content": prompt}],
            model=cfg.REFORMULATION_MODEL,
            temperature=0.0,
            max_tokens=200,
        )

    def _llm_extract_entities(self, user_msg: str, assistant_msg: str) -> list[str]:
        llm = get_llm()
        prompt = (
            f"Extract named entities (people, subjects, concepts, places) from:\n"
            f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
            "Return a comma-separated list only. Example: Newton, gravity, 1687"
        )
        raw = llm.chat(
            [{"role": "user", "content": prompt}],
            model=cfg.REFORMULATION_MODEL,
            temperature=0.0,
            max_tokens=80,
        )
        return [e.strip() for e in raw.split(",") if e.strip()]

    # ── public API ─────────────────────────────────────────────────────────────
    def get_memory(self, session_id: str):
        raw_hist  = self._redis.get(self._hkey(session_id))
        raw_sum   = self._redis.get(self._skey(session_id))
        raw_ents  = self._redis.get(self._ekey(session_id))

        history  = self._json.loads(raw_hist)  if raw_hist  else []
        summary  = raw_sum                      if raw_sum   else ""
        entities = self._json.loads(raw_ents)   if raw_ents  else []
        return history, summary, entities

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        history, summary, _ = self.get_memory(session_id)

        # append new turn
        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant",  "content": assistant_msg})

        # summarise + trim when window overflows
        if len(history) > self.MAX_TURNS * 2:
            summary = self._llm_summarize(history, summary)
            history = history[-(self.MAX_TURNS * 2):]  # keep only last N turns

        # entity extraction (async-friendly; runs synchronously here)
        entities = self._llm_extract_entities(user_msg, assistant_msg)

        pipe = self._redis.pipeline()
        pipe.set(self._hkey(session_id), self._json.dumps(history),    ex=self.SUMMARY_TTL)
        pipe.set(self._skey(session_id), summary,                       ex=self.SUMMARY_TTL)
        pipe.set(self._ekey(session_id), self._json.dumps(entities),   ex=self.SUMMARY_TTL)
        pipe.execute()

@lru_cache(maxsize=1)
def get_memory() -> MemoryService:
    return MemoryService()
