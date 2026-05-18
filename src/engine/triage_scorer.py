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
IS_VERBOSE = True
print("start")

def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "I feel hopeless and alone"
    with open("src/engine/bool_prompt.txt", "r") as f:
        PROMPT_BOOL = f.read()
    with open("src/engine/explaination_prompt.txt", "r") as f:
        PROMPT_EXPLAIN = f.read()
    close_results = query_database(query)
    print(f"Query: {query!r}\n")
    APPENDED_PROMPT_BOOL = f"{PROMPT_BOOL}\n{close_results}\nPost to classify: \n{query}"
    generated_bool_response = generate_output(LLM_MODEL, APPENDED_PROMPT_BOOL, options={"temperature": 0})
    classified_risk = classify_risk(generated_bool_response)
    APPENDED_PROMPT_EXPLAIN = f"{PROMPT_EXPLAIN}\n\nPost to explain: {query}\n\nScoring dict: {classified_risk}\n\nSimilar posts: {close_results}"
    generated_explain_response = generate_output(LLM_MODEL, APPENDED_PROMPT_EXPLAIN, options={"temperature": 0.5})
    print("START OF GENERATED RESPONSE: ")
    print(f"Risk score: {classified_risk['risk_score']}")
    print(f"Severity: {classified_risk['severity']}")
    print(f"Explaination: {generated_explain_response}")
    best_match = min(close_results, key=lambda r: r['distance'])
    print(f"Lowest distance score found is: {best_match['distance']}")
    similarity_score = 1 - best_match['distance']
    confidence = (classified_risk['risk_score'] * 0.7) + (similarity_score * 0.3)
    print(f"Final confidence score is: {confidence}")
    if(classified_risk['human_review_required'] is True):
        review_choice = input("High risk - human review required (options: confirm, override [<moderate, or low>], dismiss): ")
    if IS_VERBOSE is True:
        print("VERBOSE OUTPUT: \n")
        print(f"SIMILAR POSTS: \n")
        for i, r in enumerate(close_results, 1):
            label = "CRISIS" if r["in_crisis"] else "no crisis"
            print(f"#{i}  distance={r['distance']}  [{label}]  id={r['id']}")
            print(f"    Post: {r['post'][:120]}...")
            print(f"    Why:  {r['explanation'][:100]}...")
            print()
        print()
        print("GENERATED BOOLEAN SWITCHES: ")
        print(f"{generated_bool_response}\n")


if __name__ == "__main__":
    main()