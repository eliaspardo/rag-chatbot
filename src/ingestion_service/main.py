from typing import List
from fastapi import FastAPI, HTTPException

from src.ingestion_service.bootstrap import prepare_vector_store
from src.ingestion_service.lifespan import lifespan
from pydantic import BaseModel

from src.shared.exceptions import IngestionRequestException, NoDocumentsException
from src.shared.exceptions import (
    ChromaException,
    VectorStoreException,
    ServerSetupException,
)
from src.shared.constants import Error
import logging

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)


class IngestionRequest(BaseModel):
    documents: List[str]


class IngestionResponse(BaseModel):
    success: bool
    message: str


@app.get("/health")
def read_root():
    return {"status": "ok"}


@app.post("/ingestion/documents/", response_model=IngestionResponse)
def ingest_documents(request: IngestionRequest):
    print("Processing ingestion request...")
    try:
        app.state.vectordb = prepare_vector_store(
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
