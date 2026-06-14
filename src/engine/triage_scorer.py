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

    top_results = query_database(query, 10) # Get top 10 similar posts from the database
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
    
    confidence_score = scoring_engine.calculate_final_confidence_score(risk_profile, top_results)
    risk_profile["confidence_score"] = confidence_score
    #if risk_profile["confidence_score"] < 0.9 and risk_profile["severity"] != "low":
    #    risk_profile["human_review_required"] = True

    print("2/2: Generating risk profile...")
    APPENDED_PROMPT_EXPLAIN = f"{PROMPT_EXPLAIN}\n\nPost to explain: {query}\n\nRisk profile: {risk_profile}\n\nSimilar posts: {top_results[:3]}"
    options_explain = {
        "temperature": 0.5,
        "num_predict": 1024,
        "num_ctx": 4096
    }
    generated_explain_response = generate_output(LLM_MODEL, APPENDED_PROMPT_EXPLAIN, options=options_explain)
    # print('2/2: Completed risk profile')
    
    print("\nSTART OF GENERATED RESPONSE: ")
    print(f"Query: {query!r}")
    print(f"\nExplanation: {generated_explain_response}")
    
    print(f"\nConfidence score: {risk_profile['confidence_score']*100:.1f}%")
    print(f"Risk score: {risk_profile['risk_score']*100:.1f}%")
    print(f"Severity: {risk_profile['severity']}")
    
    print("\nFINAL VERDICT: ", end="")
    if risk_profile['in_crisis']:
        print("DANGER - IN CRISIS")
    else:
        print("SAFE - NOT IN CRISIS")
    
    
    review_choice = None
    if risk_profile['human_review_required'] and not ignore_human_review:
        print("""
High risk - human review required (options: (c)onfirm, to override enter <(h)igh/(m)oderate/(l)ow>, (d)issmiss)
Please note that override will change the following risk profile values:
    - in_crisis: will be set to True for 'h', False for 'm', 'l', and 'd'
    - risk_score: will be set to 1.0 for 'h', capped at 0.5 for 'm', 0.2 for 'l', and 0.0 for 'd'
    - severity: will be set to 'high' for 'h', 'moderate' for 'm' and 'low' for 'l' and 'd'""")
        while True:
            review_choice = input("""Enter 'c' to confirm high risk, 'h' to override to high, 'm' to override to moderate, 'l' to override to low, or 'd' to dismiss as safe: """).strip().lower()
            if review_choice == 'c':
                review_choice = "confirmed_high"
            elif review_choice == 'h':
                review_choice = "override_high"
                risk_profile['in_crisis'] = True
                risk_profile['risk_score'] = max(risk_profile['risk_score'], 1.0)
                risk_profile['severity'] = "high"
            elif review_choice == 'm':
                review_choice = "override_moderate"
                risk_profile['in_crisis'] = False
                risk_profile['risk_score'] = min(risk_profile['risk_score'], 0.5)
                risk_profile['severity'] = "moderate"
            elif review_choice == 'l':
                review_choice = "override_low"
                risk_profile['in_crisis'] = False
                risk_profile['risk_score'] = min(risk_profile['risk_score'], 0.2)
                risk_profile['severity'] = "low"
            elif review_choice == 'd':
                review_choice = "override_dismiss"
                risk_profile['in_crisis'] = False
                risk_profile['risk_score'] = 0.0
                risk_profile['severity'] = "low"
            else:
                print("Invalid choice, no changes made to risk profile.")
                continue
            break
            
    if is_verbose:
        print("\nSIMILAR POSTS: \n")
        for i, r in enumerate(top_results[:3], 1):
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
