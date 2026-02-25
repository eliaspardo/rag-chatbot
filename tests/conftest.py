import os
from datetime import datetime, timezone

import chromadb
import pytest
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion_service.vector_store_builder import get_vector_store_builder
from src.shared.env_loader import load_environment

load_environment()
EVAL_PDF_PATH = os.getenv("EVAL_PDF_PATH")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)


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
    Build an in-memory Chroma database once per test session so eval tests run
    against fresh embeddings and leave no artifacts behind.
    """
    if not EVAL_PDF_PATH:
        pytest.skip("Evals skipped: set EVAL_PDF_PATH to run these tests.")

    vector_store_builder = get_vector_store_builder()
    texts = vector_store_builder.load_pdf_text(str(EVAL_PDF_PATH))
    docs = vector_store_builder.split_text_to_docs(texts)

    # Create in-memory ChromaDB for testing
    chroma_client = chromadb.EphemeralClient()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(
        docs,
        embeddings,
        client=chroma_client,
        collection_name="eval_test_collection",
    )
    yield vectordb
    # EphemeralClient cleans up automatically
