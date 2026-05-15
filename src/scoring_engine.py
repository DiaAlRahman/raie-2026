CATEGORY_A = {
    "suicidal_ideation": 1.0,
    "selfharm_mention": 1.0,
    "indirect_suicidal_ideation": 0.8,
    "danger_to_others": 0.8,
}

CATEGORY_B = {
    "worthlessness_expressed": 0.55,
    "helplessness_expressed": 0.55,
    "hopelessness_expressed": 0.55,
}

CATEGORY_C = {
    "self_isolation": 0.1,
    "previous_conditions": 0.1,
    "previous_suicidality": 0.1,
}

def calculate_risk_score(flags):
    score = 0

    a_count = sum(flags.get(switch) is True for switch in CATEGORY_A)
    b_count = sum(flags.get(switch) is True for switch in CATEGORY_B)
    c_count = sum(flags.get(switch) is True for switch in CATEGORY_C)

    for category in [CATEGORY_A, CATEGORY_B, CATEGORY_C]:
        for switch, weight in category.items():
            if flags.get(switch) is True:
                score += weight

    score = min(score, 1.0)    # Cap score at 1.0

    high_risk = (
        a_count >= 1 or
        (b_count >= 2 and c_count == 0) or
        (b_count >= 1 and c_count >= 2)
    )

    if high_risk:
        score = max(score, 0.75)

    if score >= 0.75:
        label = "High risk"
    elif score >= 0.25:
        label = "Moderate risk"
    else:
        label = "Low risk"

    return {
        "score": round(score * 100, 2),
        "label": label,
        "human_review_required": label == "High risk",
        "category_counts": {
            "category_a": a_count,
            "category_b": b_count,
            "category_c": c_count,
        }
    }

# =========================
# TEST CASE
# =========================

test_flags = {
    "suicidal_ideation": False,
    "selfharm_mention": False,
    "indirect_suicidal_ideation": False,
    "danger_to_others": False,

    "worthlessness_expressed": True,
    "helplessness_expressed": True,
    "hopelessness_expressed": False,

    "self_isolation": False,
    "previous_conditions": False,
    "previous_suicidality": False,
}

result = calculate_risk_score(test_flags)

print(result)