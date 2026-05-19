from __future__ import annotations

import os
import sys
from pathlib import Path

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

from database.query_db import query_database
import engine.scoring_engine as scoring_engine
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


# ==========================================
# 4. MAIN PIPELINE LOGIC
# ==========================================
def run_pipeline(query: str, is_verbose: bool = False, ignore_human_review: bool = False) -> dict:
    """Executes the triage scoring pipeline on a given query string."""
    
    # Use dynamic absolute paths so the script doesn't break if run from outside the root folder
    with open(CURRENT_DIR / "bool_prompt.txt", "r") as f:
        PROMPT_BOOL = f.read()
    with open(CURRENT_DIR / "explanation_prompt.txt", "r") as f:
        PROMPT_EXPLAIN = f.read()

    top_10_results = query_database(query, 10) # Get top 10 similar posts from the database
    # print(f"Query: {query!r}\n")

    print("1/2: Generating boolean classification...")
    APPENDED_PROMPT_BOOL = f"{PROMPT_BOOL}\nPost to classify: \n{query}"
    options_bool = {
        "temperature": 0.0,
        "num_predict": 512,
        "num_ctx": 4096
    }
    generated_bool_response = generate_output(LLM_MODEL, APPENDED_PROMPT_BOOL, options=options_bool)
    risk_profile = scoring_engine.generate_profile(generated_bool_response)
    # print('1/2: Completed boolean classification')

    print("2/2: Generating risk profile...")
    APPENDED_PROMPT_EXPLAIN = f"{PROMPT_EXPLAIN}\n\nPost to explain: {query}\n\nRisk profile: {risk_profile}\n\nSimilar posts: {top_10_results[:3]}"
    options_explain = {
        "temperature": 0.5,
        "num_predict": 1024,
        "num_ctx": 4096
    }
    generated_explain_response = generate_output(LLM_MODEL, APPENDED_PROMPT_EXPLAIN, options=options_explain)
    # print('2/2: Completed risk profile')
    
    print("\nSTART OF GENERATED RESPONSE: ")
    print(f"Query: {query!r}")
    print(f"Explanation: {generated_explain_response}")
    
    confidence_score = scoring_engine.calculate_final_confidence_score(risk_profile, top_10_results)
    
    print(f"\nConfidence score: {confidence_score*100:.2f}%")
    print(f"Risk score: {risk_profile['risk_score']*100:.2f}%")
    print(f"Severity: {risk_profile['severity']}")
    
    print("\nFINAL VERDICT: ", end="")
    if risk_profile['in_crisis']:
        print("DANGER - IN CRISIS")
    else:
        print("SAFE - NOT IN CRISIS")
    
    
    review_choice = None
    if (risk_profile['human_review_required'] or confidence_score < 0.9) and not ignore_human_review:
        review_choice = input("""\nHigh risk - human review required (options: (c)onfirm, to override enter <(m)oderate/(l)ow>, (d)issmiss) 
                              enter 'c' to confirm high risk, 'm' to override to moderate, 'l' to override to low, or 'd' to dismiss as safe: """).strip().lower()
    if is_verbose:
        print("\nSIMILAR POSTS: \n")
        for i, r in enumerate(top_10_results[:3], 1):
            label = "IN CRISIS" if r["in_crisis"] else "NOT IN CRISIS"
            print(f"#{i}  distance={r['distance']:.2f}  [{label}]  id={r['id']}")
            print(f"    Post: {r['post'][:120]}...")
            print(f"    Why:  {r['explanation'][:100]}...")
            print()
        print()
        print("GENERATED BOOLEAN SWITCHES: ")
        print(f"{generated_bool_response}\n")
        
    return {
        "in_crisis": risk_profile['in_crisis'],
        "explanation": generated_explain_response,
        "risk_score": risk_profile['risk_score'],
        "severity": risk_profile['severity'],
        "confidence_score": confidence_score,
        "human_review_choice": review_choice
    }
