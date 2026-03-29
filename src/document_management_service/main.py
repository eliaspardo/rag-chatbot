"""FastAPI application for the Document Management Service."""

from typing import List
from fastapi import Depends, FastAPI, HTTPException, Response
from pydantic import ValidationError
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT
from src.document_management_service.lifespan import lifespan
from src.shared.constants import SetDocumentResult
from src.shared.exceptions import DocumentHashConflictException
from sqlalchemy.exc import SQLAlchemyError
from src.shared.models import (
    DMSDocument,
    GetDocumentStatusResponse,
    SetDocumentStatusRequest,
)
import logging
from src.document_management_service.db_client import DBClient

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)


def get_db_client():
    """FastAPI dependency that yields a DBClient backed by a new SQLAlchemy session."""
    session = app.state.Session()
    try:
        yield DBClient(session)
    finally:
        session.close()


@app.get("/health")
def health():
    """Return a simple health-check response."""
    return {"status": "ok"}


@app.get("/documents/{doc_hash}/status/", response_model=GetDocumentStatusResponse)
def get_document_status(doc_hash, db_client: DBClient = Depends(get_db_client)):
    """Retrieve the current processing status of a document by its hash."""
    logger.info("Processing get document status request...")
    try:
        doc_name = db_client.get_document_name(doc_hash)
        if not doc_name:
            raise HTTPException(status_code=404)
        status = db_client.get_document_status(doc_hash)
        return GetDocumentStatusResponse(doc_name=doc_name, status=status)
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Processing failed")


@app.put("/documents/{doc_hash}/status/", response_model=DMSDocument | None)
def put_document_status(
    doc_hash,
    request: SetDocumentStatusRequest,
    db_client: DBClient = Depends(get_db_client),
):
    """Create or update a document record with the given status; returns 201 on create, 204 on update."""
    logger.info("Processing put document status request...")
    try:
        document, result = db_client.set_document_status(
            doc_hash, request.doc_name, request.status
        )
    except DocumentHashConflictException as e:
        logger.error(e)
        raise HTTPException(status_code=409, detail="Document hash conflict")
    except (SQLAlchemyError, ValidationError) as e:
        logger.error(e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Processing failed")
    if result is SetDocumentResult.UPDATED:
        return Response(status_code=HTTP_204_NO_CONTENT)
    return Response(
        status_code=HTTP_201_CREATED,
        content=document.model_dump_json(),
        headers={"Content-Type": "application/json"},
    )


@app.get("/documents/", response_model=List[DMSDocument])
def get_documents(db_client: DBClient = Depends(get_db_client)):
    """Return all registered documents, or 204 No Content if none exist."""
    logger.info("Processing get documents request...")
    try:
        docs = db_client.get_documents()
        if not docs:
            return Response(status_code=HTTP_204_NO_CONTENT)
        return docs
    except (SQLAlchemyError, ValidationError) as e:
        logger.error(e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Processing failed")
