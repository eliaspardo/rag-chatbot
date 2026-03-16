from typing import List
from fastapi import FastAPI, HTTPException
from requests import HTTPError

from src.ingestion_service.lifespan import lifespan
from pydantic import BaseModel

from src.shared.exceptions import NoDocumentsException
import logging

logger = logging.getLogger(__name__)


app = FastAPI(lifespan=lifespan)


class IngestionRequest(BaseModel):
    documents: List[str]


class SingleIngestionRequest(BaseModel):
    document: str


class IngestionResponse(BaseModel):
    success: bool
    message: str


def get_vectordb_collection_count() -> int:
    return app.state.vector_store_builder.get_collection_count()


@app.get("/health")
def health():
    return {"status": "ok", "documents_loaded": f"{get_vectordb_collection_count()}"}


@app.post("/ingestion/documents/", response_model=IngestionResponse)
def ingest_documents(request: IngestionRequest):
    print("Processing ingestion request...")
    print("Using DMS-enabled ingestion...")
    app.state.doc_ingestor.ingest_documents(request.documents)
    return IngestionResponse(
        success=True, message="Documents processed and saved to vector store!"
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
