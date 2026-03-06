from typing import List
import os
from fastapi import FastAPI, HTTPException
from requests import HTTPError

from src.ingestion_service.bootstrap import update_vector_store
from src.ingestion_service.lifespan import lifespan
from pydantic import BaseModel

from src.shared.env_loader import load_environment
from src.shared.exceptions import IngestionRequestException, NoDocumentsException
from src.shared.exceptions import (
    ChromaException,
    VectorStoreException,
    ServerSetupException,
)
from src.shared.constants import Error
import logging

logger = logging.getLogger(__name__)

load_environment()
DMS_ENABLED = os.getenv("DMS_ENABLED", "false").lower() == "true"

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
    if DMS_ENABLED:
        print("Using DMS-enabled ingestion...")
        app.state.doc_ingestor.ingest_documents(request.documents)
    else:
        print("Using legacy ingestion...")
        try:
            update_vector_store(
                vector_store_builder=app.state.vector_store_builder,
                file_loader=app.state.file_loader,
                progress_callback=print,
                pdf_paths=request.documents,
            )
        except IngestionRequestException:
            logger.error(Error.EXCEPTION)
            raise HTTPException(
                status_code=422,
                detail="Error when processing pdfs provided in the request.",
            )
        except NoDocumentsException:
            logger.error("Error: no documents found to ingest.")
            raise HTTPException(status_code=422, detail="No documents found to ingest.")
        except (ChromaException, VectorStoreException):
            logger.error(Error.EXCEPTION)
            raise ServerSetupException()
        except Exception:
            logger.error(Error.EXCEPTION)
            raise ServerSetupException()
    return IngestionResponse(
        success=True, message="Documents processed and saved to vector store!"
    )


@app.post("/ingestion/document/", response_model=IngestionResponse)
def ingest_document(request: SingleIngestionRequest):
    print("Processing ingestion request...")
    if not DMS_ENABLED:
        raise HTTPException(
            status_code=404, detail="Function not supported in Legacy Ingestion mode"
        )
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
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Processing failed",
        )
    return IngestionResponse(
        success=True, message="Document processed and saved to vector store!"
    )
