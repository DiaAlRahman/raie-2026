"""
Task 4: Retrieval function for the Beyond Blue ChromaDB vector store.
Task 5: Distance scores are included in every returned result.

Usage:
    from src.database.query_db import query_database

    results = query_database("I feel like giving up on everything")
    for r in results:
        print(r["distance"], r["in_crisis"], r["post"])

CLI:
    python src/database/query_db.py "I feel like giving up"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

import chromadb
import httpx
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(REPO_ROOT / "chroma_db")

COLLECTION_NAME = "beyondblue_posts"
EMBEDDING_MODEL = "mxbai-embed-large"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
OLLAMA_TIMEOUT = 120.0

MAX_CHARS_FOR_EMBEDDING = 1800


class RetrievedPost(TypedDict):
    id: str
    post: str
    in_crisis: bool
    explanation: str
    distance: float


class _OllamaWithTimeout(OllamaEmbeddingFunction):
    def __init__(self, url: str, model_name: str, timeout: float = OLLAMA_TIMEOUT):
        super().__init__(url=url, model_name=model_name)
        self._session = httpx.Client(timeout=timeout)


def _get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = _OllamaWithTimeout(url=OLLAMA_URL, model_name=EMBEDDING_MODEL)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def query_database(text: str, n_results: int = 3) -> list[RetrievedPost]:
    """Embed *text* with mxbai-embed-large and return the top-n similar posts.

    Each result includes the ChromaDB cosine distance score (Task 5).
    Lower distance = more similar (range 0–2, cosine space).
    """
    collection = _get_collection()
    query_text = text[:MAX_CHARS_FOR_EMBEDDING]

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    posts: list[RetrievedPost] = []
    for doc_id, metadata, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        posts.append(
            RetrievedPost(
                id=doc_id,
                post=metadata["post"],
                in_crisis=bool(metadata["in_crisis"]),
                explanation=metadata["explanation"],
                distance=round(distance, 6),
            )
        )

    return posts


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "I feel hopeless and alone"
    print(f"Query: {query!r}\n")
    for i, r in enumerate(query_database(query), 1):
        label = "CRISIS" if r["in_crisis"] else "no crisis"
        print(f"#{i}  distance={r['distance']}  [{label}]  id={r['id']}")
        print(f"    Post: {r['post'][:120]}...")
        print(f"    Why:  {r['explanation'][:100]}...")
        print()
