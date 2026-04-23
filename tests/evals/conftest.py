import os

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


@pytest.fixture(scope="session")
def eval_test_vectordb():
    """
    Build an in-memory Chroma database once per test session so eval tests run
    against fresh embeddings and leave no artifacts behind.
    """
    if not EVAL_PDF_PATH:
        pytest.skip("Evals skipped: set EVAL_PDF_PATH to run these tests.")

    chroma_client = chromadb.EphemeralClient()

    vector_store_builder = get_vector_store_builder(chroma_client)
    texts = vector_store_builder.load_pdf_text(str(EVAL_PDF_PATH))
    docs = vector_store_builder.split_text_to_docs(texts)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(
        docs,
        embeddings,
        client=chroma_client,
        collection_name="eval_test_collection",
    )
    yield vectordb
