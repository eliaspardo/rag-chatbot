from __future__ import annotations
from collections.abc import Callable
import os
from typing import List
from src.shared.exceptions import (
    NoDocumentsException,
)
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import VectorStoreBuilder
from langchain_core.documents import Document
from src.shared.env_loader import load_environment
import logging

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[str], None]

load_environment()
PDF_PATH = os.getenv("PDF_PATH")


def process_documents(
    pdf_paths: List[str],
    file_loader: FileLoader,
    vector_store_builder: VectorStoreBuilder,
    progress: ProgressCallback,
) -> List[Document]:
    progress("🔍 Processing PDFs...")
    docs: List[Document] = []
    for file in pdf_paths:
        document = process_document(file, file_loader, vector_store_builder, progress)
        if not document:
            continue
        docs.extend(document)
    if not docs:
        raise NoDocumentsException("No documents found after splitting.")
    return docs


def process_document(
    file: str,
    file_loader: FileLoader,
    vector_store_builder: VectorStoreBuilder,
    progress: ProgressCallback,
) -> List[Document] | None:
    try:
        file_path = file_loader.load_pdf_file(file)
    except FileNotFoundError:
        logger.error(f"Error processing {file}")
        return None
    texts = vector_store_builder.load_pdf_text(file_path)
    progress(f"✀ Splitting text to docs for {file_path}")
    return vector_store_builder.split_text_to_docs(texts)
