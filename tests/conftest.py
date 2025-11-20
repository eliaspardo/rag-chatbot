import shutil
from pathlib import Path

import pytest

from src.rag_preprocessor import RAGPreprocessor

ISTQB_PDF_PATH = Path("data/ISTQB_CTAL-TM_Syllabus_v3.0.pdf")
ISTQB_DB_DIR = Path("tests/data/istqb_tm_faiss_db")


@pytest.fixture(scope="session")
def istqb_vectordb():
    """
    Build the ISTQB FAISS database once per test session so ragas tests run
    against fresh embeddings and leave no artifacts behind.
    """
    preprocessor = RAGPreprocessor()
    texts = preprocessor.load_pdf_text(str(ISTQB_PDF_PATH))
    docs = preprocessor.split_text_to_docs(texts)

    if ISTQB_DB_DIR.exists():
        shutil.rmtree(ISTQB_DB_DIR)

    preprocessor.create_vector_store(docs=docs, db_dir=str(ISTQB_DB_DIR))
    try:
        yield str(ISTQB_DB_DIR)
    finally:
        shutil.rmtree(ISTQB_DB_DIR, ignore_errors=True)
