"""
Build the ChromaDB vector database for the Beyond Blue triage project.

Merges Task 2 (collection init) and Task 3 (embedding pipeline) into one
script. Reads master_dataset.csv produced by Task 1, runs each post through
mxbai-embed-large via Ollama, and upserts the 1024-dim vectors into a
persistent ChromaDB collection.

Collection structure (per-row metadata):
    post         -- the original post text (str, full length)
    in_crisis    -- the human risk label (bool)
    explanation  -- the human reasoning behind the label (str)

The text fed to the embedding model is truncated to MAX_CHARS_FOR_EMBEDDING
(mxbai-embed-large has a 512-token context window). The full original text
is preserved in metadata['post'] so Team 3's scoring sees complete content.

Run:
    python src/database/build_db.py             # full run, resume-safe
    python src/database/build_db.py --reset     # wipe collection and start over
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

# Resolve paths against the repo root so the script works regardless of cwd.
# This file is at <repo>/src/database/build_db.py, so parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(REPO_ROOT / "chroma_db")
MASTER_CSV_PATH = str(REPO_ROOT / "data" / "master_dataset.csv")

COLLECTION_NAME = "beyondblue_posts"
EMBEDDING_MODEL = "mxbai-embed-large"

# OLLAMA_HOST is set to http://host.docker.internal:11434 by the devcontainer
# so the Python container can reach the Ollama instance running on the host.
# Outside the devcontainer (running directly on a laptop) we fall back to
# localhost.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"

BATCH_SIZE = 64

# mxbai-embed-large has a 512-token context window (~4 chars/token for English,
# so ~2000 chars). 1800 leaves headroom for tokenization edge cases. Posts
# longer than this are truncated FOR EMBEDDING ONLY; the original text is
# preserved in metadata['post'].
MAX_CHARS_FOR_EMBEDDING = 1800

METADATA_FIELDS: tuple[str, ...] = ("post", "in_crisis", "explanation")


# ---------------------------------------------------------------------------
# Collection setup (imported by src/database/query_db.py for Task 4)
# ---------------------------------------------------------------------------

def init_collection():
    """Create or open the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBEDDING_MODEL,
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_posts(collection, ids, posts, in_crisis_labels, explanations):
    """Upsert posts into the collection using the required metadata schema.

    The text passed to the embedding model is truncated to fit the model's
    context window; metadata['post'] keeps the full original text.
    """
    n = len(ids)
    if not (len(posts) == len(in_crisis_labels) == len(explanations) == n):
        raise ValueError("ids, posts, in_crisis_labels, explanations must be same length")

    posts_for_embedding = [p[:MAX_CHARS_FOR_EMBEDDING] for p in posts]

    metadatas = [
        {"post": p, "in_crisis": c, "explanation": e}
        for p, c, e in zip(posts, in_crisis_labels, explanations)
    ]

    collection.upsert(ids=ids, documents=posts_for_embedding, metadatas=metadatas)


# ---------------------------------------------------------------------------
# Pipeline (Task 3)
# ---------------------------------------------------------------------------

def load_csv(csv_path):
    """Read master_dataset.csv and coerce in_crisis to bool."""
    df = pd.read_csv(csv_path)
    if df["in_crisis"].dtype != bool:
        df["in_crisis"] = df["in_crisis"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        )
    return df


def filter_already_embedded(df, collection):
    """Drop rows whose master_post_id is already in the collection (resume-safe)."""
    existing = set(collection.get()["ids"])
    if not existing:
        return df, 0
    keep_mask = ~df["master_post_id"].isin(existing)
    skipped = int((~keep_mask).sum())
    return df[keep_mask].reset_index(drop=True), skipped


def run_pipeline(csv_path, limit=None):
    collection = init_collection()

    df = load_csv(csv_path)
    if limit:
        df = df.head(limit)

    df, skipped = filter_already_embedded(df, collection)
    if skipped:
        print(f"{skipped} rows already embedded, {len(df)} remaining")

    if len(df) == 0:
        print("Nothing to embed.")
        return

    for start in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[start : start + BATCH_SIZE]
        add_posts(
            collection,
            ids=chunk["master_post_id"].tolist(),
            posts=chunk["post"].tolist(),
            in_crisis_labels=chunk["in_crisis"].tolist(),
            explanations=chunk["explanation"].tolist(),
        )
        print(f"  {min(start + BATCH_SIZE, len(df))}/{len(df)}")

    print(f"Collection size: {collection.count()}")


def reset_collection():
    """Delete the collection so the next run rebuilds from scratch."""
    client = chromadb.PersistentClient(path=DB_PATH)
    names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in names:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted collection '{COLLECTION_NAME}'")


def main():
    parser = argparse.ArgumentParser(description="Build the ChromaDB vector database.")
    parser.add_argument("--csv", default=MASTER_CSV_PATH, help="path to master_dataset.csv")
    parser.add_argument("--limit", type=int, default=None, help="embed only first N rows")
    parser.add_argument("--reset", action="store_true", help="wipe collection before running")
    args = parser.parse_args()

    if args.reset:
        reset_collection()

    if not Path(args.csv).exists():
        raise SystemExit(
            f"CSV not found: {args.csv}\n"
            f"Download data.zip from the team OneDrive and place master_dataset.csv "
            f"at data/master_dataset.csv in the repo root."
        )

    run_pipeline(args.csv, limit=args.limit)


if __name__ == "__main__":
    main()
