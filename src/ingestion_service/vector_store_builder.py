import os
import re
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_docling.loader import DoclingLoader, ExportType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
from src.shared.exceptions import ChromaException, VectorStoreException
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


class VectorStoreBuilder:
    def __init__(self, chroma_client=None):
        self.chroma_client = chroma_client or chromadb.HttpClient(
            host=CHROMA_HOST, port=CHROMA_PORT
        )

    def collection_has_documents(self):
        try:
            collection = self.chroma_client.get_collection(CHROMA_COLLECTION)
            return collection.count() > 0
        except Exception:
            return False

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

    # --- Embed and Store in Chroma ---
    def create_vector_store(
        self,
        docs: list[Document],
        model_name: str = EMBEDDING_MODEL,
    ) -> Chroma:
        try:
            logger.debug("👉 Initializing HuggingFaceEmbeddings")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            logger.debug(f"👉 Creating Chroma DB with {len(docs)} docs")
            try:
                vectordb = Chroma.from_documents(
                    docs,
                    embeddings,
                    client=self.chroma_client,
                    collection_name=CHROMA_COLLECTION,
                )
                logger.debug("✅ Chroma.from_documents completed successfully")
            except ValueError as exception:
                raise ChromaException(
                    f"Invalid documents for Chroma: {exception}"
                ) from exception
            except RuntimeError as exception:
                raise ChromaException(
                    f"Chroma creation failed: {exception}"
                ) from exception
            logger.debug("✅ Vector store creation successful")
            return vectordb
        except (ChromaException, VectorStoreException):
            raise
        except Exception as exception:
            raise VectorStoreException(
                f"Error creating Vector Store: {exception}"
            ) from exception


class LegacyVectorStoreBuilder(VectorStoreBuilder):
    def __init__(self, chroma_client=None):
        super().__init__(chroma_client)

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


class DoclingVectorStoreBuilder(VectorStoreBuilder):
    def __init__(self, chroma_client=None):
        super().__init__(chroma_client)
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


def get_vector_store_builder(chroma_client=None) -> VectorStoreBuilder:
    if RAG_PREPROCESSOR == "docling":
        return DoclingVectorStoreBuilder(chroma_client)
    if RAG_PREPROCESSOR == "legacy":
        return LegacyVectorStoreBuilder(chroma_client)
    else:
        logger.warning("RAG_PREPROCESSOR not defined! Defaulting to legacy.")
        return LegacyVectorStoreBuilder(chroma_client)
