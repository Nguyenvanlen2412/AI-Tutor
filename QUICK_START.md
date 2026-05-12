# Quick Start Guide – AI Tutor

## Required Services

Your AI Tutor requires **4 external services** to run. Open 4 terminal windows and run these commands:

### Terminal 1: Ollama (LLM + Safety)
```bash
ollama serve
```
Then in a 5th terminal, pull the required models:
```bash
ollama pull gemma3:4b
ollama pull llama-guard3:1b
```

### Terminal 2: Qdrant (Vector Database)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Terminal 3: Redis Stack (Semantic Cache)
```bash
docker run -p 6379:6379 redis/redis-stack-server
```

### Terminal 4: Zep (Conversation Memory)
```bash
docker compose up -d
---

## Run the AI Tutor

Once all 4 services are running, in your 5th terminal:

### Interactive Chat
```bash
python main.py
```

### Single Query
```bash
python main.py --query "Hi, explain Python decorators"
```

### Voice Input (requires .wav file)
```bash
python main.py --voice lecture.wav --output voice
```

---

## Environment Setup

If you haven't created the virtual environment yet:

```bash
# Create and activate venv
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy config
copy .env.example .env
# Edit .env with your settings if needed
```

---

## Minimal Test (without services)

To test the pipeline without running all services:
```bash
python main.py --query "test"
```

You'll see errors for missing services, but you can verify:
- ✅ Graph construction works
- ✅ LangGraph routing is correct
- ✅ State management functions

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `No connection could be made... 11434` | Start Ollama: `ollama serve` |
| `No module named 'qdrant_client'` | Run: `pip install -r requirements.txt` |
| `Connection refused... 6379` | Start Redis: `docker run -p 6379:6379 redis/redis-stack-server` |
| `Connection refused... 8000` | Start Zep: `docker run -p 8000:8000 -e ZEP_STORE_BASE_DIR=/tmp/zep getzep/zep:latest` |

---

## Docker not available?

If you don't have Docker for Qdrant/Redis/Zep, you can:
1. Install locally via native installers
2. Use cloud-hosted versions (provide API URLs in `.env`)
3. Comment out retrieval/caching for text-only demo

For a quick demo without Docker, you could modify the code to skip memory/vector store errors gracefully.
