"""
Driver .py file that implements each stage of the pipeline
         
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

from ollama import Client
from ollama import chat
from ollama import ChatResponse


from database.query_db import query_database
from scoring_engine import classify_risk
from engine.prompts import generate_boolswitch_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(REPO_ROOT / "chroma_db")

COLLECTION_NAME = "beyondblue_posts"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "mistral"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
OLLAMA_TIMEOUT = 120.0

MAX_CHARS_FOR_EMBEDDING = 1800


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "I feel hopeless and alone"
    with open("src/engine/bool_prompt.txt", "r") as f:
        PROMPT_BOOL = f.read()
    print(PROMPT_BOOL)
    close_results = query_database(query)
    print(f"Query: {query!r}\n")
    for i, r in enumerate(close_results, 1):
        label = "CRISIS" if r["in_crisis"] else "no crisis"
        print(f"#{i}  distance={r['distance']}  [{label}]  id={r['id']}")
        print(f"    Post: {r['post'][:120]}...")
        print(f"    Why:  {r['explanation'][:100]}...")
        print()
    print(close_results)
    APPENDED_PROMPT_BOOL = f"{PROMPT_BOOL}\n{close_results}\nPost to classify: \n{query}"
    print(APPENDED_PROMPT_BOOL)
    print("START OF BOOLSWITCH PROMPT OUTPUT:")
    print(generate_boolswitch_prompt(LLM_MODEL, APPENDED_PROMPT_BOOL))

if __name__ == "__main__":
    main()