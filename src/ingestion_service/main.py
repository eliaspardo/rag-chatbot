from typing import List
from fastapi import FastAPI, HTTPException
from requests import HTTPError

from src.ingestion_service.lifespan import lifespan
from pydantic import BaseModel

from src.shared.exceptions import NoDocumentsException
import logging

from src.shared.models import DMSDocument

logger = logging.getLogger(__name__)


app = FastAPI(lifespan=lifespan)


class IngestionRequest(BaseModel):
    documents: List[str]


class SingleIngestionRequest(BaseModel):
    document: str


class IngestionResponse(BaseModel):
    success: bool
    message: str


class DocumentResult(BaseModel):
    document: str
    success: bool
    error: str | None = None


class BatchIngestionResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: List[DocumentResult]


def get_vectordb_collection_count() -> int:
    """
    Get the current number of collections stored in the application's vector store.
    
    Returns:
        int: The number of collections in the vector store.
    """
    return app.state.vector_store_builder.get_collection_count()


def get_dms_documents() -> List[DMSDocument]:
    """
    Fetch documents from the DMS client attached to the application's document ingestor.
    
    Returns:
        List[DMSDocument]: List of retrieved DMSDocument objects. Returns an empty list if an error occurs while fetching.
    """
    try:
        documents = app.state.doc_ingestor.dms_client.get_documents()
        return documents
    except Exception as e:
        print(f"Error getting documents from DMS: {e}")
        return []


@app.get("/health")
def health():
    """
    Report application health and counts of documents loaded in the vector store and DMS.
    
    Returns:
        dict: A mapping with keys:
            - "status": the service health status, normally "ok".
            - "documents_loaded_in_vector_store": the vector store collection count as a string.
            - "documents_loaded_in_dms": a list of DMS documents converted to dicts via `model_dump()`.
    """
    documents = [doc.model_dump() for doc in get_dms_documents()]

    return {
        "status": "ok",
        "documents_loaded_in_vector_store": f"{get_vectordb_collection_count()}",
        "documents_loaded_in_dms": documents,
    }


@app.post("/ingestion/documents/", response_model=BatchIngestionResponse)
def ingest_documents(request: IngestionRequest):
    print("Processing ingestion request...")
    print("Using DMS-enabled ingestion...")
    results = app.state.doc_ingestor.ingest_documents(request.documents)
    succeeded = sum(1 for r in results if r.success)
    return BatchIngestionResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=[
            DocumentResult(document=r.document, success=r.success, error=r.error)
            for r in results
        ],
    )


@app.post("/ingestion/document/", response_model=IngestionResponse)
def ingest_document(request: SingleIngestionRequest):
    print("Processing ingestion request...")
    try:
        print("Using DMS-enabled ingestion...")
        app.state.doc_ingestor.ingest_document(request.document)
    except NoDocumentsException:
        raise HTTPException(
            status_code=404,
            detail="File or documents not found",
        )
    except HTTPError:
        raise HTTPException(
            status_code=503,
            detail="Error calling DMS",
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=500,
            detail="Processing failed",
        )
    return IngestionResponse(
        success=True, message="Document processed and saved to vector store!"
    )
