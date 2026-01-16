import shutil
import os
from pathlib import Path

import pytest

from src.core.rag_preprocessor import RAGPreprocessor
from src.env_loader import load_environment

load_environment()
EVAL_PDF_PATH = os.getenv("EVAL_PDF_PATH")
EVAL_DB_DIR_ENV = os.getenv("EVAL_DB_DIR")
EVAL_DB_DIR = Path(EVAL_DB_DIR_ENV) if EVAL_DB_DIR_ENV else None


@pytest.fixture(scope="session")
def ragas_test_vectordb():
    """
    Build the Ragas test FAISS database once per test session so ragas tests run
    against fresh embeddings and leave no artifacts behind.
    """
    if not EVAL_PDF_PATH or not EVAL_DB_DIR:
        pytest.skip(
            "Ragas eval skipped: set EVAL_PDF_PATH and EVAL_DB_DIR to run these tests."
        )

    preprocessor = RAGPreprocessor()
    texts = preprocessor.load_pdf_text(str(EVAL_PDF_PATH))
    docs = preprocessor.split_text_to_docs(texts)

    if EVAL_DB_DIR.exists():
        shutil.rmtree(EVAL_DB_DIR)

    preprocessor.create_vector_store(docs=docs, db_dir=str(EVAL_DB_DIR))
    try:
        yield str(EVAL_DB_DIR)
    finally:
        shutil.rmtree(EVAL_DB_DIR, ignore_errors=True)
