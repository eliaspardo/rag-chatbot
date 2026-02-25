from __future__ import annotations
from collections.abc import Callable
from langchain_community.vectorstores import Chroma
from src.shared.exceptions import ConfigurationException
from src.inference_service.core.rag_preprocessor import RAGPreprocessor

ProgressCallback = Callable[[str], None]


def prepare_vector_store(
    rag_preprocessor: RAGPreprocessor,
    progress_callback: ProgressCallback | None = None,
) -> Chroma:
    progress = progress_callback or (lambda _: None)

    if not rag_preprocessor.collection_has_documents():
        raise ConfigurationException("Vector store has no documents!!")

    progress("📶 Loading vector store.")
    return rag_preprocessor.load_vector_store()
