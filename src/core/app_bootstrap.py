from __future__ import annotations
from collections.abc import Callable
import os
from typing import List
from langchain_community.vectorstores import Chroma
from src.shared.exceptions import ConfigurationException, NoDocumentsException
from src.core.file_loader import FileLoader
from src.core.rag_preprocessor import RAGPreprocessor
from langchain.schema import Document
from src.shared.env_loader import load_environment

ProgressCallback = Callable[[str], None]

load_environment()
PDF_PATH = os.getenv("PDF_PATH")


def prepare_vector_store(
    rag_preprocessor: RAGPreprocessor,
    file_loader: FileLoader,
    progress_callback: ProgressCallback | None = None,
) -> Chroma:
    progress = progress_callback or (lambda _: None)

    if not rag_preprocessor.collection_has_documents():
        progress("\n🔍 Loading PDFs...")
        docs: List[Document] = []
        if not PDF_PATH or not PDF_PATH.strip():
            raise ConfigurationException("PDF_PATH is empty.")
        try:
            pdf_paths = [p.strip() for p in PDF_PATH.split(",") if p.strip()]
        except Exception:
            raise ConfigurationException("Error when reading PDF_PATH.")
        for file in pdf_paths:
            file_path = file_loader.load_pdf_file(file)
            texts = rag_preprocessor.load_pdf_text(file_path)
            progress(f"✀ Splitting text to docs for {file_path}")
            docs.extend(rag_preprocessor.split_text_to_docs(texts))

        if not docs:
            raise NoDocumentsException("No documents found after splitting.")

        progress("🏭 Creating vector store.")
        rag_preprocessor.create_vector_store(docs)
        progress("✅ Vector DB created and saved.")
    else:
        progress("📦 Using existing vector store.")

    progress("📶 Loading vector store.")
    return rag_preprocessor.load_vector_store()
