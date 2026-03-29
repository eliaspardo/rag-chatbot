"""Bootstrap helpers for the inference service startup."""

from __future__ import annotations
from collections.abc import Callable
from langchain_community.vectorstores import Chroma
from src.inference_service.core.vector_store_loader import VectorStoreLoader

ProgressCallback = Callable[[str], None]


def prepare_vector_store(
    vector_store_loader: VectorStoreLoader,
    progress_callback: ProgressCallback | None = None,
    max_retries: int = 30,
) -> Chroma:
    """Load the vector store, logging a warning if no documents are present."""
    progress = progress_callback or (lambda _: None)

    progress("Checking vector store for documents...")
    if not vector_store_loader.collection_has_documents():
        progress("Vector store has no documents! Proceeding without documents.")

    progress("📶 Loading vector store.")
    return vector_store_loader.load_vector_store()
