# flake8: noqa

EVALUATION_STEPS = [
    "Check if the actual output directly addresses the input question (not a different question or topic).",
    "Verify the response is coherent and logically structured (not rambling, contradictory, or mixing unrelated topics).",
    "Ensure the answer avoids 'confident but empty' language (vague statements that sound authoritative but lack substance).",
    "Check that the response maintains appropriate professional style and organization (clear, well-structured, easy to follow).",
    "Score = 10 if the answer is relevant, coherent, well-organized, and substantive; score = 0 if it fails any of these criteria.",
    "In your reason: identify any relevance issues, coherence problems, empty language, or organizational issues, then justify the score.",
]

METADATA = {
    "version": "v1",
    "date": "2025-01-25",
    "description": "Reasoning for relevance and coherence.",
    "notes": "Initial version",
}
