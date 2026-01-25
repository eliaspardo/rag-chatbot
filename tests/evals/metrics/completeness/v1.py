# flake8: noqa

EVALUATION_STEPS = [
    "Identify all structural elements, steps, definitions, or items required by the input to fully answer the query.",
    "Compare the actual output against these requirements to determine what is present and what is missing.",
    "Use the expected output as a reference for what a complete answer should include.",
    "Score = 10 if all required elements are present; score = 0 if any major structural parts or essential information are omitted.",
    "In your reason, list all required elements, mark which are present vs. missing, and explain the final decision.",
]

METADATA = {
    "version": "v1",
    "date": "2025-01-24",
    "description": "Completeness focused on the ground truth.",
    "notes": "Initial version",
}
