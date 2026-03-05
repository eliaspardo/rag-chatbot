from __future__ import annotations
from collections.abc import Callable
import os
from typing import List
from src.shared.exceptions import (
    IngestionRequestException,
    NoDocumentsException,
    ConfigurationException,
)
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import VectorStoreBuilder
from langchain.schema import Document
from src.shared.env_loader import load_environment
import logging

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[str], None]

load_environment()
PDF_PATH = os.getenv("PDF_PATH")


def prepare_vector_store(
    vector_store_builder: VectorStoreBuilder,
    file_loader: FileLoader,
    progress_callback: ProgressCallback | None = None,
):
    progress = progress_callback or (lambda _: None)

    # Ran when starting server up: use existing store or seed with PDF_PATH
    if vector_store_builder.collection_has_documents():
        progress("📦 Using existing vector store.")
        return
    else:
        if not PDF_PATH or not PDF_PATH.strip():
            logger.error("Error seeding vector store with PDF_PATH: PDF_PATH is empty!")
            raise NoDocumentsException()
        logger.info("Seeding vector store with PDF_PATH.")
    try:
        clean_pdf_paths = [p.strip() for p in PDF_PATH.split(",") if p.strip()]
    except Exception:
        raise ConfigurationException("Error when reading PDFs provided.")
    try:
        docs = process_documents(
            clean_pdf_paths, file_loader, vector_store_builder, progress
        )
    except NoDocumentsException:
        logger.error("Error seeding vector store: no documents found after splitting!")
        raise NoDocumentsException()
    progress("🏭 Creating vector store.")
    vector_store_builder.add_documents_to_vector_store(docs)
    progress("✅ Vector DB created and saved.")


def update_vector_store(
    vector_store_builder: VectorStoreBuilder,
    file_loader: FileLoader,
    progress_callback: ProgressCallback | None = None,
    pdf_paths: List[str] = None,
):
    progress = progress_callback or (lambda _: None)

    try:
        clean_pdf_paths = [p.strip() for p in pdf_paths if p.strip()]
    except Exception:
        raise IngestionRequestException("Error when reading PDFs provided.")
    try:
        docs = process_documents(
            clean_pdf_paths, file_loader, vector_store_builder, progress
        )
    except NoDocumentsException:
        logger.error("Error seeding vector store: no documents found after splitting!")
        raise NoDocumentsException()
    progress("🏭 Updating vector store.")
    vector_store_builder.add_documents_to_vector_store(docs)
    progress("✅ Vector DB updated and saved.")


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
