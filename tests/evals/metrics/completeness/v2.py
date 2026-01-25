# flake8: noqa

EVALUATION_STEPS = [
    "Extract the required key elements from the expected output (treat them as a checklist).",
    "Check whether each required element is present in the actual output (exact match or clear paraphrase is acceptable).",
    "Do NOT penalize additional details or extra information, as long as all required elements are present.",
    "Score = 10 if all required elements are present; score = 0 if any required element is missing.",
    "In your reason: list each required element and mark it as Present or Missing, then justify the score.",
]

METADATA = {
    "version": "v2",
    "date": "2025-01-25",
    "description": "Completeness focused on the ground truth.",
    "notes": "Focused check based on expected output",
}
