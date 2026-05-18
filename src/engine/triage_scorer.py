from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

# ==========================================
# 1. PATH RESOLUTION (CRITICAL FOR IMPORTS)
# ==========================================
# Get the directory this file is in (src/engine)
CURRENT_DIR = Path(__file__).resolve().parent
# Get the src/ directory
SRC_DIR = CURRENT_DIR.parent
# Get the root of the repo (ProjectRaie)
REPO_ROOT = SRC_DIR.parent

# Add src/ to the Python path so it can find database/ and scoring_engine.py
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ==========================================
# 2. LOCAL IMPORTS 
# ==========================================
import chromadb
import httpx
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from ollama import Client, chat, ChatResponse

from database.query_db import query_database
from engine.scoring_engine import classify_risk
from engine.prompts import generate_output


# ==========================================
# 3. GLOBALS 
# ==========================================
DB_PATH = str(REPO_ROOT / "chroma_db")
COLLECTION_NAME = "beyondblue_posts"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "mistral"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://172.17.0.1:11434")
OLLAMA_URL = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
OLLAMA_TIMEOUT = 120.0

MAX_CHARS_FOR_EMBEDDING = 1800
IS_VERBOSE = True


# ==========================================
# 4. MAIN PIPELINE LOGIC
# ==========================================
def run_pipeline(query: str):
    """Executes the triage scoring pipeline on a given query string."""
    
    # Use dynamic absolute paths so the script doesn't break if run from outside the root folder
    with open(CURRENT_DIR / "bool_prompt.txt", "r") as f:
        PROMPT_BOOL = f.read()
    with open(CURRENT_DIR / "explaination_prompt.txt", "r") as f:
        PROMPT_EXPLAIN = f.read()

    close_results = query_database(query)
    print(f"Query: {query!r}\n")

    print("1/2: Generating boolean classification...")
    # APPENDED_PROMPT_BOOL = f"{PROMPT_BOOL}\n{close_results}\nPost to classify: \n{query}"
    generated_bool_response = generate_output(LLM_MODEL, PROMPT_BOOL, options={"temperature": 0})
    classified_risk = classify_risk(generated_bool_response)
    print('1/2: Completed boolean classification.')

    print("\n2/2: Generating risk explanation...")
    APPENDED_PROMPT_EXPLAIN = f"{PROMPT_EXPLAIN}\n\nPost to explain: {query}\n\nScoring dict: {classified_risk}\n\nSimilar posts: {close_results}"
    generated_explain_response = generate_output(LLM_MODEL, APPENDED_PROMPT_EXPLAIN, options={"temperature": 0.5})
    print('2/2: Completed risk explanation.')
    
    print("\nSTART OF GENERATED RESPONSE: ")
    print(f"Risk score: {classified_risk['risk_score']}")
    print(f"Severity: {classified_risk['severity']}")
    print(f"Explanation: {generated_explain_response}")
    
    best_match = min(close_results, key=lambda r: r['distance'])
    print(f"Lowest distance score found is: {best_match['distance']}")
    similarity_score = 1 - best_match['distance']
    confidence = (classified_risk['risk_score'] * 0.7) + (similarity_score * 0.3)
    print(f"Final confidence score is: {confidence}")
    
    if classified_risk['human_review_required']:
        review_choice = input("High risk - human review required (options: confirm, override [<moderate, or low>], dismiss): ")
        
    if IS_VERBOSE:
        print("VERBOSE OUTPUT: \n")
        print("SIMILAR POSTS: \n")
        for i, r in enumerate(close_results, 1):
            label = "CRISIS" if r["in_crisis"] else "no crisis"
            print(f"#{i}  distance={r['distance']}  [{label}]  id={r['id']}")
            print(f"    Post: {r['post'][:120]}...")
            print(f"    Why:  {r['explanation'][:100]}...")
            print()
        print()
        print("GENERATED BOOLEAN SWITCHES: ")
        print(f"{generated_bool_response}\n")

def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "I feel hopeless and alone"
    run_pipeline(query)

if __name__ == "__main__":
    main()