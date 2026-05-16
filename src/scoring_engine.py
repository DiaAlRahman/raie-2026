import json


CATEGORY_A = {
    "suicidal_ideation": 1.0,
    "selfharm_mention": 1.0,
    "indirect_suicidal_ideation": 0.8,
    "danger_to_others": 0.8
}

CATEGORY_B = {
    "worthlessness_expressed": 0.55,
    "helplessness_expressed": 0.55,
    "hopelessness_expressed": 0.55
}

CATEGORY_C = {
    "self_isolation": 0.1,
    "previous_conditions": 0.1,
    "previous_suicidality": 0.1
}


ALL_CATEGORIES = {
    **CATEGORY_A,
    **CATEGORY_B,
    **CATEGORY_C
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

    for field in ALL_CATEGORIES:
        cleaned[field] = bool(data.get(field, False))

    return cleaned


def count_true(data, category):
    return sum(1 for field in category if data.get(field, False))


def calculate_score(data):
    score = 0

    for field, weight in ALL_CATEGORIES.items():
        if data.get(field, False):
            score += weight

    return min(score, 1.0)


def classify_risk(llm_output):
    data = parse_llm_output(llm_output)
    cleaned = clean_llm_output(data)

    a_count = count_true(cleaned, CATEGORY_A)
    b_count = count_true(cleaned, CATEGORY_B)
    c_count = count_true(cleaned, CATEGORY_C)

    score = calculate_score(cleaned)

    high_risk_rule_triggered = (
        a_count >= 1
        or (b_count >= 2 and c_count == 0)
        or (b_count >= 1 and c_count >= 2)
    )

    if high_risk_rule_triggered:
        score = max(score, 0.75)
        severity = "high"
        human_review_required = True

    elif score >= 0.75:
        severity = "high"
        human_review_required = True

    elif score >= 0.25:
        severity = "moderate"
        human_review_required = False

    else:
        severity = "low"
        human_review_required = False

    return {
        "risk_score": round(score, 2),
        "severity": severity,
        "human_review_required": human_review_required,
        "category_counts": {
            "A": a_count,
            "B": b_count,
            "C": c_count
        },
        "indicators": cleaned
    }