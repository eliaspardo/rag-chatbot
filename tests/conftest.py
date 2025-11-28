import shutil
import os
from pathlib import Path

import pytest

from src.rag_preprocessor import RAGPreprocessor
from src.env_loader import load_environment

load_environment()
RAGAS_PDF_PATH = os.getenv("RAGAS_PDF_PATH")
RAGAS_DB_DIR_ENV = os.getenv("RAGAS_DB_DIR")
RAGAS_DB_DIR = Path(RAGAS_DB_DIR_ENV) if RAGAS_DB_DIR_ENV else None


@pytest.fixture(scope="session")
def ragas_test_vectordb():
    """
    Build the Ragas test FAISS database once per test session so ragas tests run
    against fresh embeddings and leave no artifacts behind.
    """
    if not RAGAS_PDF_PATH or not RAGAS_DB_DIR:
        pytest.skip(
            "Ragas eval skipped: set RAGAS_PDF_PATH and RAGAS_DB_DIR to run these tests."
        )

    preprocessor = RAGPreprocessor()
    texts = preprocessor.load_pdf_text(str(RAGAS_PDF_PATH))
    docs = preprocessor.split_text_to_docs(texts)

    if RAGAS_DB_DIR.exists():
        shutil.rmtree(RAGAS_DB_DIR)

    preprocessor.create_vector_store(docs=docs, db_dir=str(RAGAS_DB_DIR))
    try:
        yield str(RAGAS_DB_DIR)
    finally:
        shutil.rmtree(RAGAS_DB_DIR, ignore_errors=True)
