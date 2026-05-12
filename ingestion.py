"""
ingestion.py – one-shot document ingestion pipeline.

Usage:
    python ingestion.py --docs ./docs --collection ai_tutor_docs

Supports: PDF, DOCX, Markdown, HTML, plain text.

Pipeline:
  1. Discover files in --docs directory (recursive).
  2. Load each file with the appropriate LangChain document loader.
  3. Split into chunks with RecursiveCharacterTextSplitter.
  4. Embed each chunk with BGE-M3.
  5. Upsert to Qdrant with source metadata.
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import cfg
from services import get_embedder, get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Qdrant collection bootstrap ───────────────────────────────────────────────

def ensure_collection(client, collection_name: str, vector_size: int) -> None:
    from qdrant_client.models import Distance, VectorParams

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{collection_name}' (dim={vector_size}).")
    else:
        logger.info(f"Collection '{collection_name}' already exists – skipping creation.")


# ── File discovery ────────────────────────────────────────────────────────────

LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc":  Docx2txtLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm":  UnstructuredHTMLLoader,
    ".txt":  TextLoader,
}


def load_documents(docs_dir: str) -> List[Document]:
    """Recursively load all supported files from *docs_dir*."""
    docs: List[Document] = []
    root = Path(docs_dir)

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        loader_cls = LOADER_MAP.get(suffix)
        if loader_cls is None:
            logger.debug(f"Skipping unsupported file: {path}")
            continue
        try:
            loader = loader_cls(str(path))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata.setdefault("source", str(path))
            docs.extend(loaded)
            logger.info(f"Loaded {len(loaded)} page(s) from {path}")
        except Exception as exc:
            logger.warning(f"Failed to load {path}: {exc}")

    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks (size={cfg.CHUNK_SIZE}, overlap={cfg.CHUNK_OVERLAP}).")
    return chunks


# ── Embedding + Upsert ────────────────────────────────────────────────────────

def ingest_chunks(chunks: List[Document], collection_name: str, batch_size: int = 64) -> None:
    from qdrant_client.models import PointStruct
    from qdrant_client import QdrantClient

    embedder = get_embedder()
    client = QdrantClient(
        url=cfg.QDRANT_URL,
        api_key=cfg.QDRANT_API_KEY or None,
        timeout=60,
    )
    ensure_collection(client, collection_name, cfg.QDRANT_VECTOR_SIZE)

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.page_content for c in batch]

        vectors = embedder.embed(texts)          # (N, 1024) ndarray

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i].tolist(),
                payload={
                    "text":   texts[i],
                    "source": batch[i].metadata.get("source", "unknown"),
                    **{k: v for k, v in batch[i].metadata.items() if k != "source"},
                },
            )
            for i in range(len(batch))
        ]

        client.upsert(collection_name=collection_name, points=points)
        logger.info(
            f"Upserted batch {start // batch_size + 1} "
            f"({min(start + batch_size, total)}/{total} chunks)."
        )

    logger.info("Ingestion complete.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant for AI Tutor.")
    parser.add_argument("--docs",       default=cfg.DOCS_DIR,           help="Directory containing documents.")
    parser.add_argument("--collection", default=cfg.QDRANT_COLLECTION,  help="Qdrant collection name.")
    parser.add_argument("--batch-size", type=int, default=64,           help="Embedding batch size.")
    args = parser.parse_args()

    if not os.path.isdir(args.docs):
        logger.error(f"Document directory not found: {args.docs}")
        return

    docs   = load_documents(args.docs)
    chunks = split_documents(docs)
    ingest_chunks(chunks, args.collection, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
