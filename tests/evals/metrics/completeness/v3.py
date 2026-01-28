# flake8: noqa

EVALUATION_STEPS = [
    "Extract the required key elements from the expected output (treat them as a checklist of concepts/facts to verify).",
    "Check whether each required element is present in the actual output - accept exact matches, paraphrases, grammatical variations, or answers that contain the required element as part of a more complete/precise term.",
    "Focus on whether the CONCEPT or FACT is present, not whether the exact wording matches. If the actual answer provides more specific or complete information that includes the required element, consider it present.",
    "Do NOT penalize additional details, context, or explanations as long as all required elements are present.",
    "Score = 10 if all required elements are present; score = 0 if any required element is missing.",
    "In your reason: list each required element and mark it as Present or Missing, then justify the score.",
]

METADATA = {
    "version": "v3",
    "date": "2025-01-27",
    "description": "Completeness focused on the ground truth.",
    "notes": "More lenient on wording, not penalizing overprecision.",
}
