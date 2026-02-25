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


class RAGPreprocessor:
    def __init__(self):
        self.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    def collection_has_documents(self):
        try:
            collection = self.chroma_client.get_collection(CHROMA_COLLECTION)
            return collection.count() > 0
        except Exception:
            return False

    # --- Load Vector Storage for Retrieval ---
    def load_vector_store(self, model_name: str = EMBEDDING_MODEL) -> Chroma:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectordb = Chroma(
            embedding_function=embeddings,
            client=self.chroma_client,
            collection_name=CHROMA_COLLECTION,
        )
        return vectordb


def get_rag_preprocessor() -> RAGPreprocessor:
    return RAGPreprocessor()
