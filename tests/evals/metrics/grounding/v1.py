# flake8: noqa

EVALUATION_STEPS = [
    "Extract factual claims from the actual output.",
    "Verify each claim is supported by the retrieval context.",
    "Use the expected output only to confirm the core facts you should be checking.",
    "Score = 10 if all claims are grounded in context; score = 0 if any claim is ungrounded.",
    "In your reason, list grounded claims, ungrounded claims (if any), and the final decision.",
]

METADATA = {
    "version": "v1",
    "date": "2025-01-19",
    "description": "Grounding focused on retrieval context support.",
    "notes": "Works with EVAL_MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
}
