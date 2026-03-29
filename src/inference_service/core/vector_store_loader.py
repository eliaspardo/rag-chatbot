"""Vector store loader for the inference service, providing read-only ChromaDB access."""

import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from src.shared.env_loader import load_environment

logger = logging.getLogger(__name__)

load_environment()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RAG_PREPROCESSOR = os.getenv("RAG_PREPROCESSOR", "legacy")
DOCLING_EXPORT_TYPE = os.getenv("DOCLING_EXPORT_TYPE", "doc_chunks")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")


class VectorStoreLoader:
    """Loads an existing ChromaDB collection as a LangChain Chroma vector store."""

    def __init__(self, chroma_client=None):
        self.chroma_client = chroma_client or chromadb.HttpClient(
            host=CHROMA_HOST, port=CHROMA_PORT
        )

    def collection_has_documents(self):
        """Return True if the ChromaDB collection contains at least one document."""
        try:
            collection_count = self.get_collection_count()
            return collection_count > 0
        except Exception:
            return False

    def get_collection_count(self) -> int:
        """Return the number of documents in the ChromaDB collection, or 0 on error."""
        try:
            return self.chroma_client.get_collection(CHROMA_COLLECTION).count()
        except Exception:
            return 0

    def load_vector_store(self, model_name: str = EMBEDDING_MODEL) -> Chroma:
        """Instantiate and return a Chroma vector store backed by HuggingFace embeddings."""
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectordb = Chroma(
            embedding_function=embeddings,
            client=self.chroma_client,
            collection_name=CHROMA_COLLECTION,
        )
        return vectordb


def get_vector_store_loader(chroma_client=None) -> VectorStoreLoader:
    """Instantiate and return a VectorStoreLoader with the given or default ChromaDB client."""
    return VectorStoreLoader(chroma_client)
