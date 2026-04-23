from datetime import datetime, timezone

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--run-name",
        action="store",
        default=None,
        help="Custom name for MLflow run (default: deepeval-YYYY-MM-DD-HH-MM-SS)",
    )
    parser.addoption(
        "--question-id",
        action="store",
        type=int,
        default=None,
        help="Only run the dataset entry with the given question_id",
    )


@pytest.fixture(scope="session")
def run_name(request):
    """
    Fixture to provide run name for MLflow runs.
    Uses --run-name command-line option if provided, otherwise defaults to
    deepeval-{timestamp}.
    """
    custom_run_name = request.config.getoption("--run-name")
    if custom_run_name:
        return custom_run_name
    return f"deepeval-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}"


@pytest.fixture(scope="session")
def run_specific_question_id(request) -> int | None:
    """
    Fixture to run specific question_id.
    Uses --question-id command-line option if provided.
    """
    question_id = request.config.getoption("--question-id")
    if question_id is None:
        return None
    return int(question_id)
