from __future__ import annotations
from collections.abc import Callable
from langchain_community.vectorstores import Chroma
from src.shared.exceptions import ServerSetupException
from src.inference_service.core.vector_store_loader import VectorStoreLoader
import time

ProgressCallback = Callable[[str], None]


def prepare_vector_store(
    vector_store_loader: VectorStoreLoader,
    progress_callback: ProgressCallback | None = None,
    max_retries: int = 30,
) -> Chroma:
    progress = progress_callback or (lambda _: None)

    retries = 0
    while not vector_store_loader.collection_has_documents():
        progress("Checking vector store for documents...")
        if retries >= max_retries:
            raise ServerSetupException("Vector store has no documents!!")
        time.sleep(2)
        retries += 1

    progress("📶 Loading vector store.")
    return vector_store_loader.load_vector_store()
