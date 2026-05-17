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
from engine.prompts import generate_output

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = str(REPO_ROOT / "chroma_db")

COLLECTION_NAME = "beyondblue_posts"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "mistral"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
OLLAMA_TIMEOUT = 120.0

MAX_CHARS_FOR_EMBEDDING = 1800
print("start")

def main():
    print("start")
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "I feel hopeless and alone"
    with open("src/engine/bool_prompt.txt", "r") as f:
        PROMPT_BOOL = f.read()
    with open("src/engine/explaination_prompt.txt", "r") as f:
        PROMPT_EXPLAIN = f.read()
    
    print(PROMPT_BOOL)
    print(PROMPT_EXPLAIN)
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
    generated_bool_response = generate_output(LLM_MODEL, APPENDED_PROMPT_BOOL, options={"temperature": 0})
    print(generated_bool_response)
    classified_risk = classify_risk(generated_bool_response)
    print(classified_risk)
    # I would just need query, classify_risk(generated response), and then close_results, and it could go off that
    APPENDED_PROMPT_EXPLAIN = f"{PROMPT_EXPLAIN}\n\nPost to explain: {query}\n\nScoring dict: {classified_risk}\n\nSimilar posts: {close_results}"
    print("START OF APPENDED PROMPT EXPLAIN")
    print(APPENDED_PROMPT_EXPLAIN)
    generated_explain_response = generate_output(LLM_MODEL, APPENDED_PROMPT_EXPLAIN, options={"temperature": 0.5})
    print("START OF GENERATED EXPLAIN RESPONSE: ")
    print(generated_explain_response)

if __name__ == "__main__":
    main()