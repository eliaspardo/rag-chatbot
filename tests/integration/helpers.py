import os
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from responses import RequestsMock


def seed_chromadb_documents(
    chroma_client: chromadb.HttpClient,
    collection_name: str,
    texts: list[str],
    metadatas: list[dict] | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """
    Seed a ChromaDB collection with the given text documents and their embeddings for test setup.
    
    Parameters:
        collection_name (str): Target ChromaDB collection name.
        texts (list[str]): Text chunks to store as documents.
        metadatas (list[dict] | None): Optional metadata objects aligned with `texts`; if omitted, default metadata
            of the form `{"source": "test_doc_{i}"}` is created for each text.
        embedding_model (str): HuggingFace embedding model name to use (default: "sentence-transformers/all-MiniLM-L6-v2").
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


def extract_doc_name(document: str) -> str:
    """
    Extracts a filename-like basename from a URL or filesystem path.
    
    Parameters:
        document (str): A URL or path string. If a URL is provided, its path component is used; otherwise the input is treated as a path.
    
    Returns:
        str: The final path component (basename) of the resolved path.
    """
    parsed = urlparse(document)
    path = parsed.path if parsed.scheme else document
    return os.path.basename(path)


def debug_requests_mock_calls(mock_dms: RequestsMock) -> None:
    # Print all captured calls
    """
    Prints a concise summary of all HTTP requests and their responses recorded by a RequestsMock.
    
    Parameters:
        mock_dms (RequestsMock): The `responses.RequestsMock` instance whose captured `calls` will be printed. Each entry's request method, URL, body, and response status code are displayed.
    """
    print(f"\n📋 Total HTTP calls made: {len(mock_dms.calls)}")
    for i, call in enumerate(mock_dms.calls, 1):
        print(f"\nCall {i}:")
        print(f"  Method: {call.request.method}")
        print(f"  URL: {call.request.url}")
        print(f"  Body: {call.request.body}")
        print(f"  Response Status: {call.response.status_code}")
