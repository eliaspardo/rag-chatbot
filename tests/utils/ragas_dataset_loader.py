import json
import os
from pathlib import Path
from typing import List, Tuple

# Default path to the golden set used by RAGAS evaluations (env overrides allowed)
DEFAULT_GOLDEN_SET_PATH = Path(os.getenv("EVAL_GOLDEN_SET_PATH"))


class GoldenSetValidationError(ValueError):
    """Raised when the golden set JSON does not match the expected schema."""


def _validate_entry(index: int, entry: dict) -> Tuple[str, str]:
    if not isinstance(entry, dict):
        raise GoldenSetValidationError(
            f"Invalid item at index {index}: expected an object with 'question' and 'ground_truth' fields."
        )

    question = entry.get("question")
    ground_truth = entry.get("ground_truth")

    if not isinstance(question, str) or not question.strip():
        raise GoldenSetValidationError(
            f"Invalid or missing 'question' at index {index}: expected a non-empty string."
        )
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise GoldenSetValidationError(
            f"Invalid or missing 'ground_truth' at index {index}: expected a non-empty string."
        )

    return question, ground_truth


def load_golden_set_dataset(
    path: Path | str | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Load and validate the golden set dataset for RAGAS tests.

    Args:
        path: Optional custom path to the dataset JSON file. Defaults to tests/data/golden_set.json
              (overridable via EVAL_GOLDEN_SET_PATH).

    Returns:
        A tuple of (questions, ground_truths) lists.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        GoldenSetValidationError: If the file contents fail schema validation.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    dataset_path = Path(path) if path else DEFAULT_GOLDEN_SET_PATH

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Golden set dataset not found at {dataset_path}. "
            "Provide a JSON array of {'question': str, 'ground_truth': str} objects."
        )

    with dataset_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise GoldenSetValidationError(
            "Golden set JSON must be a non-empty list of objects with 'question' and 'ground_truth'."
        )

    questions: List[str] = []
    ground_truths: List[str] = []
    for index, entry in enumerate(data):
        question, ground_truth = _validate_entry(index, entry)
        questions.append(question)
        ground_truths.append(ground_truth)

    return questions, ground_truths
