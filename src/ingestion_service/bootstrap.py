from __future__ import annotations
from collections.abc import Callable
import os
from typing import List
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import VectorStoreBuilder
from langchain_core.documents import Document
from src.shared.env_loader import load_environment
import logging

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[str], None]

load_environment()
PDF_PATH = os.getenv("PDF_PATH")


def process_document(
    file: str,
    file_loader: FileLoader,
    vector_store_builder: VectorStoreBuilder,
    progress: ProgressCallback,
) -> List[Document] | None:
    """
    Load a PDF by identifier, extract its text, and split that text into a list of Document objects.
    
    Parameters:
        file (str): Path or identifier of the PDF to load.
        file_loader (FileLoader): Component responsible for resolving/loading the PDF file.
        vector_store_builder (VectorStoreBuilder): Component used to extract text from the PDF and split it into documents.
        progress (ProgressCallback): Callback invoked with progress messages.
    
    Returns:
        List[Document] | None: List of documents produced from the PDF text, or `None` if the PDF file was not found.
    """
    try:
        file_path = file_loader.load_pdf_file(file)
    except FileNotFoundError:
        logger.error(f"Error processing {file}")
        return None
    texts = vector_store_builder.load_pdf_text(file_path)
    progress(f"✀ Splitting text to docs for {file_path}")
    return vector_store_builder.split_text_to_docs(texts)
