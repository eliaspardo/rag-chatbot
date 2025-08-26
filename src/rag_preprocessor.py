import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from constants import Error
from exceptions import FaissException, VectorStoreException
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- CONFIGURATION FROM .ENV ---
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)
PDF_PATH = os.getenv("PDF_PATH")
DB_DIR = os.getenv("DB_DIR", "faiss_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


class RAGPreprocessor:
    # --- Extract and Split Text ---
    def load_pdf_text(path: str = PDF_PATH) -> list[str]:
        doc = fitz.open(path)
        texts = [page.get_text() for page in doc]
        return texts

    # --- Chunk Text into Documents ---
    def split_text_to_docs(
        self,
        texts: list[str],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        full_text = "\n".join(texts)
        chunks = splitter.split_text(full_text)

        # Filter out empty chunks
        valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        docs = [Document(page_content=chunk) for chunk in valid_chunks]

        logger.debug(
            f"📄 Created {len(docs)} documents from {len(valid_chunks)} valid chunks"
        )
        return docs

    # --- Embed and Store in FAISS ---
    def create_vector_store(
        self,
        docs: list[Document],
        db_dir: str = DB_DIR,
        model_name: str = EMBEDDING_MODEL,
    ) -> FAISS:

        # Add this before creating the vector store
        if os.path.exists(db_dir):
            logger.debug(f"🧹 Removing existing directory: {db_dir}")
            shutil.rmtree(db_dir)

        try:
            logger.debug("👉 Initializing HuggingFaceEmbeddings")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            logger.debug(f"👉 Creating FAISS DB at '{db_dir}' with {len(docs)} docs")
            try:
                vectordb = FAISS.from_documents(docs, embeddings)
                logger.debug(
                    "✅ FAISS.from_documents completed successfully",
                    flush=True,
                )
            except Exception as exception:
                raise FaissException(Error.FAISS_EXCEPTION.value) from exception
            logger.debug("👉 Persisting FAISS DB")
            vectordb.save_local(db_dir)
            logger.debug("✅ Vector store creation successful")
            return vectordb
        except Exception as exception:
            raise VectorStoreException(Error.VECTOR_EXCEPTION.value) from exception

    # --- Load Vector Storage for Retrieval ---
    def load_vector_store(
        self, db_dir: str = DB_DIR, model_name: str = EMBEDDING_MODEL
    ) -> FAISS:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectordb = FAISS.load_local(
            db_dir, embeddings, allow_dangerous_deserialization=True
        )
        return vectordb
