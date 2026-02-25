from __future__ import annotations
from collections.abc import Callable
from langchain_community.vectorstores import Chroma
from src.shared.exceptions import ConfigurationException
from src.inference_service.core.vector_store_loader import VectorStoreLoader

ProgressCallback = Callable[[str], None]


def prepare_vector_store(
    vector_store_loader: VectorStoreLoader,
    progress_callback: ProgressCallback | None = None,
) -> Chroma:
    progress = progress_callback or (lambda _: None)

    if not vector_store_loader.collection_has_documents():
        raise ConfigurationException("Vector store has no documents!!")

    progress("📶 Loading vector store.")
    return vector_store_loader.load_vector_store()
