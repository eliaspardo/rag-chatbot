import shutil
import os
from pathlib import Path


from dotenv import load_dotenv

import pytest

from src.rag_preprocessor import RAGPreprocessor

# Load environment variables
load_dotenv()
RAGAS_PDF_PATH = os.getenv("RAGAS_PDF_PATH")
RAGAS_DB_DIR = Path(os.getenv("RAGAS_DB_DIR"))


@pytest.fixture(scope="session")
def ragas_test_vectordb():
    """
    Build the Ragas test FAISS database once per test session so ragas tests run
    against fresh embeddings and leave no artifacts behind.
    """
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
