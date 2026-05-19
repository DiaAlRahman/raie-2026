import json
import re


WEIGHTS = {
    "crisis_action": 1.0,           # Immediate escalation
    "severe_distress": 0.4,         # Moderate indicator
    "social_withdrawal": 0.2,       # Behavioral context
    "historical_vulnerability": 0.1 # Historical context multiplier
}

def parse_llm_output(llm_output):
    if isinstance(llm_output, dict):
        return llm_output
        
    if isinstance(llm_output, str):
        # Step 1: Strip out markdown blocks that LLMs love to use
        cleaned = llm_output.replace('```json', '').replace('```', '').strip()
        
        # Step 2: Fix the exact bug you just found! If it forgot to close the dictionary, add the bracket.
        if cleaned.startswith('{') and not cleaned.endswith('}'):
            print("\n[DEBUG] AI forgot the closing bracket. Auto-fixing...")
            cleaned += '\n}'
        
        # Step 3: Use regex to find everything between the first { and the last }
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        
        if match:
            extracted_json = match.group(0)
            try:
                return json.loads(extracted_json)
            except json.JSONDecodeError as e:
                # If it fails for ANY other reason, print exactly what it typed
                print(f"\n[DEBUG] JSON DECODE ERROR: {e}")
                print(f"[DEBUG] RAW LLM OUTPUT WAS:\n{llm_output}\n")
                raise ValueError("LLM output is not valid JSON")
        else:
            print(f"\n[DEBUG] NO BRACKETS FOUND IN:\n{llm_output}\n")
            raise ValueError("No JSON brackets found in output")
            
    raise TypeError("LLM output must be a JSON string or dictionary")

def clean_llm_output(data):
    cleaned_output = {}
    for field in WEIGHTS.keys():
        # Ensure it safely defaults to False if the LLM missed it
        cleaned_output[field] = bool(data.get(field, False))
    return cleaned_output

def calculate_score(data):
    score = 0
    for field, weight in WEIGHTS.items():
        if data.get(field, False):
            score += weight
    return min(score, 1)

def generate_profile(llm_output):
    data = parse_llm_output(llm_output)
    cleaned_output = clean_llm_output(data)
    
    score = calculate_score(cleaned_output)

    # If they are taking crisis action, instantly flag high risk
    in_crisis = False
    if cleaned_output["crisis_action"]:
        score = max(score, 0.8)
        severity = "high"
        human_review_required = True
        in_crisis = True
    elif score >= 0.7:
        severity = "high"
        human_review_required = True
        in_crisis = True
    elif score >= 0.3:
        severity = "moderate"
        human_review_required = False
    else:
        severity = "low"
        human_review_required = False

    if in_crisis:
        initial_confidence_score = score
    else:
        initial_confidence_score = 1 - score

    return {
        "risk_score": round(score, 2),
        "severity": severity,
        "human_review_required": human_review_required,
        "in_crisis": in_crisis,
        "indicators": cleaned_output,
        "initial_confidence_score": initial_confidence_score
    }
    
def calculate_final_confidence_score(risk_profile, top_similar_posts):
    total_weight = 0.0
    agree_weight = 0.0
    
    # 1. Calculate neighborhood agreement
    for post in top_similar_posts:
        # Convert distance to similarity (higher is better)
        similarity = 1.0 - post['distance']
        total_weight += similarity
        
        # If the historical post's crisis label matches the LLM's crisis label, they agree!
        if post['in_crisis'] == risk_profile['in_crisis']:
            agree_weight += similarity
            
    # Protect against division by zero just in case the database returns nothing
    if total_weight == 0:
        neighborhood_agreement = 0.5 
    else:
        neighborhood_agreement = agree_weight / total_weight
        
    # 2. Blend the LLM's internal certainty with the historical agreement
    llm_certainty = risk_profile['initial_confidence_score']
    
    # Weigh the LLM's logic slightly heavier (70%) than historical precedent (30%)
    final_confidence = (0.7 * llm_certainty) + (0.3 * neighborhood_agreement)
    
    return final_confidence