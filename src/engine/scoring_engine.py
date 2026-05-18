import json

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
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            raise ValueError("LLM output is not valid JSON")
    raise TypeError("LLM output must be a JSON string or dictionary")

def clean_llm_output(data):
    cleaned = {}
    for field in WEIGHTS.keys():
        # Ensure it safely defaults to False if the LLM missed it
        cleaned[field] = bool(data.get(field, False))
    return cleaned

def calculate_score(data):
    score = 0.0
    for field, weight in WEIGHTS.items():
        if data.get(field, False):
            score += weight
    return min(score, 1.0)

def classify_risk(llm_output):
    data = parse_llm_output(llm_output)
    cleaned = clean_llm_output(data)
    
    score = calculate_score(cleaned)

    # If they are taking crisis action, instantly flag high risk
    if cleaned["crisis_action"]:
        score = max(score, 0.8)
        severity = "high"
        human_review_required = True
    elif score >= 0.7:
        severity = "high"
        human_review_required = True
    elif score >= 0.3:
        severity = "moderate"
        human_review_required = False
    else:
        severity = "low"
        human_review_required = False

    return {
        "risk_score": round(score, 2),
        "severity": severity,
        "human_review_required": human_review_required,
        "indicators": cleaned
    }