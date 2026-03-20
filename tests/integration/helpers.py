from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb


def seed_chromadb_documents(
    chroma_client: chromadb.HttpClient,
    collection_name: str,
    texts: list[str],
    metadatas: list[dict] | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """
    Directly seed ChromaDB with test documents.

    Bypasses the full ingestion flow - useful for test setup.

    Args:
        chroma_client: ChromaDB HTTP client pointing to testcontainer
        collection_name: Collection name (from integration_env)
        texts: List of text chunks to add
        metadatas: Optional metadata for each text chunk
        embedding_model: Model for embeddings (should match app config)
    """
    if metadatas is None:
        metadatas = [{"source": f"test_doc_{i}"} for i in range(len(texts))]

    # Create documents
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadatas)
    ]

    # Create embeddings and add to ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        client=chroma_client,
        collection_name=collection_name,
    )
