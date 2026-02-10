from __future__ import annotations
from collections.abc import Callable
import os
from typing import List
from langchain_community.vectorstores import FAISS
from src.core.exceptions import NoDocumentsException
from src.core.file_loader import FileLoader
from src.core.rag_preprocessor import DB_DIR, RAGPreprocessor
from langchain.schema import Document
from src.env_loader import load_environment

ProgressCallback = Callable[[str], None]

load_environment()
PDF_PATH = os.getenv("PDF_PATH")


def prepare_vector_store(
    rag_preprocessor: RAGPreprocessor,
    file_loader: FileLoader,
    progress_callback: ProgressCallback | None = None,
    db_dir: str = DB_DIR,
) -> FAISS:
    progress = progress_callback or (lambda _: None)

    if not os.path.exists(db_dir):
        progress("\n🔍 Loading PDF...")
        docs: List[Document] = []
        pdf_paths = [p.strip() for p in PDF_PATH.split(",") if p.strip()]
        for file in pdf_paths:
            file_path = file_loader.load_pdf_file(file)
            texts = rag_preprocessor.load_pdf_text(file_path)
            progress(f"Splitting text to docs for {file_path}")
            docs.append(rag_preprocessor.split_text_to_docs(texts))

            if not docs:
                raise NoDocumentsException("No documents found after splitting.")

        progress("Creating vector store.")
        rag_preprocessor.create_vector_store(docs, db_dir=db_dir)
        progress("✅ Vector DB created and saved.")
    else:
        progress("📦 Using existing vector store.")

    progress("📶 Loading vector store.")
    return rag_preprocessor.load_vector_store(db_dir=db_dir)
