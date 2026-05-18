# AI Tutor – Self-Hosted LangGraph Pipeline

A fully local, privacy-first AI tutor with voice support, RAG retrieval,
semantic caching, conversation memory, and multi-layer safety.

---

## Architecture

```
Voice/Text Input
       │
  [Silero VAD]  ←─ detects speech in audio stream
       │
  [Whisper STT] ←─ transcribes speech to text (self-hosted)
       │
  [Llama Guard 3-1B-INT4]  ←─ INPUT safety check
       │
  [Gemma 3B / Mistral 7B]  ←─ query reformulation
       │
  [BGE-M3 Embedder]  ←─ embed rewritten query
       │
  ┌────┴─────────────────────────┐
  │  Redis Semantic Cache        │ ←─ cache HIT → skip LLM
  └────┬─────────────────────────┘
       │ cache MISS
  [Qdrant Hybrid Search]  ←─ dense vector retrieval
       │
  [BGE-Reranker-v2-m3]   ←─ cross-encoder re-ranking
       │
  [Redis Memory]  ←─ conversation history + auto-summary + entities
       │
  [Core LLM – Gemma / custom]  ←─ response generation (via Ollama)
       │
  [Llama Guard 3-1B-INT4]  ←─ OUTPUT safety check (regen loop)
       │
  [Kokoro TTS / Zalo TTS]  ←─ text-to-speech (if voice output)
       │
  [Redis]  ←─ cache + memory update
       │
     Response
```

---

## Prerequisites – services to run first

| Service | Default port | How to run |
|---------|-------------|-----------|
| **Ollama** | 11434 | `ollama serve` then `ollama pull gemma3:4b` and `ollama pull llama-guard3:1b` |
| **Qdrant** | 6333 | `docker run -p 6333:6333 qdrant/qdrant` |
| **Redis Stack** | 6379 | `docker run -p 6379:6379 redis/redis-stack-server` |

---

## Quick start

```
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and edit config
cp .env.example .env   # set API keys, model names, etc.

# 4. Ingest documents
python ingestion.py --docs ./docs --collection ai_tutor_docs

# 5. Run interactive chat
python main.py

# One-shot text query
python main.py --query "Giải thích định lý Pythagoras"

#start server
uvicorn server:app --host 0.0.0.0 --port 8080 --reload
http://localhost:8080
```

---
## Environment variables (`.env`)

```dotenv
# Whisper
WHISPER_MODEL=base
WHISPER_LANGUAGE=en
WHISPER_DEVICE=cpu

# Ollama models
OLLAMA_BASE_URL=http://localhost:11434
LLAMA_GUARD_MODEL=llama-guard3:1b

# Embeddings
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
EMBEDDING_DEVICE=cpu

# Qdrant
QDRANT_COLLECTION=ai_tutor_docs

# Redis
REDIS_URL=redis://localhost:6379
CACHE_SIMILARITY_THRESHOLD=0.93
CACHE_TTL_SECONDS=3600


# TTS
TTS_BACKEND=kokoro          # kokoro | zalo | viettts
KOKORO_VOICE=af_heart
ZALO_TTS_API_KEY=           # only needed for zalo backend

# Safety
MAX_REGENERATIONS=3
```
