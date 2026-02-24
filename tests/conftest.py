import shutil
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.core.rag_preprocessor import get_rag_preprocessor
from src.env_loader import load_environment

load_environment()
EVAL_PDF_PATH = os.getenv("EVAL_PDF_PATH")
EVAL_DB_DIR_ENV = os.getenv("EVAL_DB_DIR")
EVAL_DB_DIR = Path(EVAL_DB_DIR_ENV) if EVAL_DB_DIR_ENV else None


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


@pytest.fixture(scope="session")
def eval_test_vectordb():
    """
    Build the Eval test Chroma database once per test session so eval tests run
    against fresh embeddings and leave no artifacts behind.
    """
    if not EVAL_PDF_PATH or not EVAL_DB_DIR:
        pytest.skip(
            "Evals skipped: set EVAL_PDF_PATH and EVAL_DB_DIR to run these tests."
        )

    preprocessor = get_rag_preprocessor()
    texts = preprocessor.load_pdf_text(str(EVAL_PDF_PATH))
    docs = preprocessor.split_text_to_docs(texts)

    if EVAL_DB_DIR.exists():
        shutil.rmtree(EVAL_DB_DIR)

    preprocessor.create_vector_store(docs=docs, db_dir=str(EVAL_DB_DIR))
    try:
        yield str(EVAL_DB_DIR)
    finally:
        shutil.rmtree(EVAL_DB_DIR, ignore_errors=True)
