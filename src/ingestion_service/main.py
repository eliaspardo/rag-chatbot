"""FastAPI application for the ingestion service."""

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
    """Request body for batch document ingestion."""

    documents: List[str]


class SingleIngestionRequest(BaseModel):
    """Request body for single document ingestion."""

    document: str


class IngestionResponse(BaseModel):
    """Response for a single document ingestion operation."""

    success: bool
    message: str


class DocumentResult(BaseModel):
    """Result for a single document within a batch ingestion."""

    document: str
    success: bool
    error: str | None = None


class BatchIngestionResponse(BaseModel):
    """Response for a batch document ingestion operation."""

    total: int
    succeeded: int
    failed: int
    results: List[DocumentResult]


def get_vectordb_collection_count() -> int:
    """Return the number of documents currently stored in the vector store."""
    return app.state.vector_store_builder.get_collection_count()


def get_dms_documents() -> List[DMSDocument]:
    """Fetch all documents registered in the Document Management Service."""
    try:
        documents = app.state.doc_ingestor.dms_client.get_documents()
        return documents
    except Exception as e:
        logger.error(f"Error getting documents from DMS: {e}")
        return []


@app.get("/health")
def health():
    """Return service health status including vector store and DMS document counts."""
    documents = [doc.model_dump() for doc in get_dms_documents()]

    return {
        "status": "ok",
        "documents_loaded_in_vector_store": f"{get_vectordb_collection_count()}",
        "documents_loaded_in_dms": documents,
    }


@app.post("/ingestion/documents/", response_model=BatchIngestionResponse)
def ingest_documents(request: IngestionRequest):
    """Ingest a batch of documents into the vector store via DMS."""
    logger.info("Processing ingestion request...")
    logger.info("Using DMS-enabled ingestion...")
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
    """Ingest a single document into the vector store via DMS."""
    logger.info("Processing ingestion request...")
    try:
        logger.info("Using DMS-enabled ingestion...")
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
