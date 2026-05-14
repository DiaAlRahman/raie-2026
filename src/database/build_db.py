"""
Build the ChromaDB vector database for the Beyond Blue triage project.

Task 2 (collection init) and Task 3 (embedding pipeline)

Collection structure:
    post         -- the original post text (str)
    in_crisis    -- the human risk label (bool)
    explanation  -- the human reasoning behind the label (str)

The text fed to the embedding model is truncated to MAX_CHARS_FOR_EMBEDDING
(mxbai-embed-large has a 512-token context window). The full original text
is preserved in metadata['post'].

Run:
    python src/database/build_db.py             # full run
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = str(REPO_ROOT / "chroma_db")
MASTER_CSV_PATH = str(REPO_ROOT / "data" / "master_dataset.csv")

COLLECTION_NAME = "beyondblue_posts"
EMBEDDING_MODEL = "mxbai-embed-large"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"

BATCH_SIZE = 64
MAX_CHARS_FOR_EMBEDDING = 1800

METADATA_FIELDS: tuple[str, ...] = ("post", "in_crisis", "explanation")


def init_collection():
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
    n = len(ids)
    if not (len(posts) == len(in_crisis_labels) == len(explanations) == n):
        raise ValueError("Wrong length for attributes")

    posts_for_embedding = [p[:MAX_CHARS_FOR_EMBEDDING] for p in posts]

    metadatas = [
        {"post": p, "in_crisis": c, "explanation": e}
        for p, c, e in zip(posts, in_crisis_labels, explanations)
    ]

    collection.upsert(ids=ids, documents=posts_for_embedding, metadatas=metadatas)


# -------------------
# Task 3
# ------------------------

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    if df["in_crisis"].dtype != bool:
        df["in_crisis"] = df["in_crisis"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        )
    return df


def filter_already_embedded(df, collection):
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
        print(f"{skipped} rows already embedded.")

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
