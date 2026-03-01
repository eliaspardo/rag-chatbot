from __future__ import annotations
from collections.abc import Callable
import os
from typing import List
from src.shared.exceptions import IngestionRequestException, NoDocumentsException
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
    pdf_paths: List[str] = None,
):
    progress = progress_callback or (lambda _: None)

    # Ran on requests:If pdf paths are provided in the request,
    if pdf_paths:
        try:
            clean_paths = [p.strip() for p in pdf_paths if p.strip()]
        except Exception:
            raise IngestionRequestException(
                "Error when reading pdf paths provided in the request."
            )
        try:
            docs = process_documents(
                clean_paths, file_loader, vector_store_builder, progress
            )
        except NoDocumentsException:
            logger.error(
                "Error seeding vector store from request: no documents found after splitting!"
            )
            raise NoDocumentsException()
        progress("🏭 Creating vector store.")
        vector_store_builder.create_vector_store(docs)
        progress("✅ Vector DB created and saved.")
    # Ran when starting server up: use existing store or seed with PDF_PATH
    else:
        if vector_store_builder.collection_has_documents():
            progress("📦 Using existing vector store.")
        else:
            progress("🔍 Loading PDFs...")
            if not PDF_PATH or not PDF_PATH.strip():
                logger.error(
                    "Error seeding vector store with PDF_PATH: PDF_PATH is empty!"
                )
                raise NoDocumentsException()
            try:
                pdf_paths = [p.strip() for p in PDF_PATH.split(",") if p.strip()]
            except Exception:
                logger.error(
                    "Error seeding vector store with PDF_PATH: error reading PDF_PATH!"
                )
                raise NoDocumentsException()
            try:
                docs = process_documents(
                    pdf_paths, file_loader, vector_store_builder, progress
                )
            except NoDocumentsException:
                logger.error(
                    "Error seeding vector store with PDF_PATH: no documents found after splitting!"
                )
                raise NoDocumentsException()
            progress("🏭 Creating vector store.")
            vector_store_builder.create_vector_store(docs)
            progress("✅ Vector DB created and saved.")


def process_documents(
    pdf_paths: List[str],
    file_loader: FileLoader,
    vector_store_builder: VectorStoreBuilder,
    progress: ProgressCallback,
) -> List[Document]:
    docs: List[Document] = []
    for file in pdf_paths:
        try:
            file_path = file_loader.load_pdf_file(file)
        except FileNotFoundError:
            logger.error(f"Error processing {file}")
            continue
        texts = vector_store_builder.load_pdf_text(file_path)
        progress(f"✀ Splitting text to docs for {file_path}")
        docs.extend(vector_store_builder.split_text_to_docs(texts))

    if not docs:
        raise NoDocumentsException("No documents found after splitting.")
    return docs
