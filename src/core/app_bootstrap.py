from __future__ import annotations
from collections.abc import Callable
import os
from langchain_community.vectorstores import FAISS
from src.core.exceptions import NoDocumentsException
from src.core.rag_preprocessor import DB_DIR, RAGPreprocessor

ProgressCallback = Callable[[str], None]


def prepare_vector_store(
    rag_preprocessor: RAGPreprocessor,
    progress_callback: ProgressCallback | None = None,
    db_dir: str = DB_DIR,
) -> FAISS:
    progress = progress_callback or (lambda _: None)

    if not os.path.exists(db_dir):
        progress("\n🔍 Loading PDF...")
        texts = rag_preprocessor.load_pdf_text()
        progress("Splitting text to docs.")
        docs = rag_preprocessor.split_text_to_docs(texts)

        if not docs:
            raise NoDocumentsException("No documents found after splitting.")

        progress("Creating vector store.")
        rag_preprocessor.create_vector_store(docs, db_dir=db_dir)
        progress("✅ Vector DB created and saved.")
    else:
        progress("📦 Using existing vector store.")

    progress("📶 Loading vector store.")
    return rag_preprocessor.load_vector_store(db_dir=db_dir)
