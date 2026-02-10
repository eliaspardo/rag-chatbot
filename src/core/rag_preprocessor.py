import os
import re
import shutil
from langchain_community.vectorstores import FAISS
from langchain_docling.loader import DoclingLoader, ExportType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
from src.core.exceptions import FaissException, VectorStoreException
import logging
from src.env_loader import load_environment

logger = logging.getLogger(__name__)

load_environment()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)
DB_DIR = os.getenv("DB_DIR", "faiss_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RAG_PREPROCESSOR = os.getenv("RAG_PREPROCESSOR", "legacy")
DOCLING_EXPORT_TYPE = os.getenv("DOCLING_EXPORT_TYPE", "doc_chunks")


class RAGPreprocessor:
    # --- Extract and Split Text ---
    def load_pdf_text(self, path) -> list[Document]:
        logger.error("No implementation for load_pdf_text")
        raise NotImplementedError

    # --- Chunk Text into Documents ---
    def split_text_to_docs(
        self,
        docs: list[Document],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> list[Document]:
        logger.error("No implementation for split_text_to_docs")
        raise NotImplementedError

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
                logger.debug("✅ FAISS.from_documents completed successfully")
            except ValueError as exception:
                raise FaissException(
                    f"Invalid documents for FAISS: {exception}"
                ) from exception
            except RuntimeError as exception:
                raise FaissException(
                    f"FAISS creation failed: {exception}"
                ) from exception
            logger.debug("👉 Persisting FAISS DB")
            vectordb.save_local(db_dir)
            logger.debug("✅ Vector store creation successful")
            return vectordb
        except Exception as exception:
            raise VectorStoreException(
                f"Error creating Vector Store: {exception}"
            ) from exception

    # --- Load Vector Storage for Retrieval ---
    def load_vector_store(
        self, db_dir: str = DB_DIR, model_name: str = EMBEDDING_MODEL
    ) -> FAISS:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectordb = FAISS.load_local(
            db_dir, embeddings, allow_dangerous_deserialization=True
        )
        return vectordb


class LegacyRAGPreprocessor(RAGPreprocessor):
    # --- Extract and Split Text ---
    def load_pdf_text(self, path: str) -> list[Document]:
        try:
            with fitz.open(path) as doc:
                texts = [Document(page.get_text()) for page in doc]
            return texts
        except Exception as e:
            raise Exception(f"Error reading PDF file {path}: {str(e)}")

    # --- Chunk Text into Documents ---
    def split_text_to_docs(
        self,
        docs: list[Document],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        full_text = "\n".join([doc.page_content for doc in docs])
        chunks = splitter.split_text(full_text)

        # Filter out empty chunks
        valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        docs = [Document(page_content=chunk) for chunk in valid_chunks]

        logger.debug(
            f"📄 Created {len(docs)} documents from {len(valid_chunks)} valid chunks"
        )
        return docs


class DoclingRAGPreprocessor(RAGPreprocessor):
    def __init__(self):
        self.EXPORT_TYPE = DOCLING_EXPORT_TYPE
        self.SECTION_HEADING_RE = re.compile(
            r"^(?P<num>\d+(?:\.\d+)*)(?:\s+)(?P<title>.+)$"
        )

    # --- Extract and Split Text ---
    def load_pdf_text(self, path: str) -> list[Document]:
        try:
            loader = DoclingLoader(
                file_path=path,
                export_type=self.EXPORT_TYPE,
            )
            docs = loader.load()
        except Exception as e:
            raise Exception(f"Error reading PDF file {path}: {str(e)}")
        return docs

    # --- Chunk Text into Documents ---
    def split_text_to_docs(
        self,
        docs: list[Document],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> list[Document]:
        if self.EXPORT_TYPE == ExportType.DOC_CHUNKS:
            # splits = docs
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=[
                    "\n\n## ",
                    "\n\n### ",
                    "\n\n",
                    "\n",
                    " ",
                ],  # Respect structure but allow merging
            )
            splits = splitter.split_documents(docs)
        elif self.EXPORT_TYPE == ExportType.MARKDOWN:
            splits = self.split_with_fallback(docs)
        else:
            raise ValueError(f"Unexpected export type: {self.EXPORT_TYPE}")
        return splits

    def split_by_numbered_headings(self, text: str) -> list[Document]:
        lines = text.splitlines()
        sections = []
        current = {"heading": None, "lines": []}

        def flush():
            if current["lines"]:
                content = "\n".join(current["lines"]).strip()
                if content:
                    sections.append(
                        Document(
                            page_content=content,
                            metadata=(
                                {"section": current["heading"]}
                                if current["heading"]
                                else {}
                            ),
                        )
                    )

        for line in lines:
            m = self.SECTION_HEADING_RE.match(line.strip())
            if m:
                flush()
                current = {
                    "heading": f'{m.group("num")} {m.group("title")}',
                    "lines": [line],
                }
            else:
                current["lines"].append(line)

        flush()
        return sections

    def split_with_fallback(self, docs: list[Document]) -> list[Document]:
        # Join docling markdown/doc chunks into one text block for section splitting
        full_text = "\n\n".join(d.page_content for d in docs)
        sections = self.split_by_numbered_headings(full_text)

        # Add section context BEFORE secondary split
        for doc in sections:
            if "section" in doc.metadata:
                doc.page_content = (
                    f"Section: {doc.metadata['section']}\n\n{doc.page_content}"
                )

        # Optional secondary split for very long sections
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        refined = []
        for doc in sections:
            chunks = splitter.split_text(doc.page_content)
            if len(chunks) <= 1:
                refined.append(doc)
            else:
                refined.extend(
                    Document(page_content=c, metadata=doc.metadata.copy())
                    for c in chunks
                )
        return refined


def get_rag_preprocessor() -> RAGPreprocessor:
    if RAG_PREPROCESSOR == "docling":
        return DoclingRAGPreprocessor()
    if RAG_PREPROCESSOR == "legacy":
        return LegacyRAGPreprocessor()
    else:
        logger.warning("RAG_PREPROCESSOR not defined! Defaulting to legacy.")
        return LegacyRAGPreprocessor()
