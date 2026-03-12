from typing import List
from fastapi import FastAPI, HTTPException, Response
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

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents/{doc_hash}/status/", response_model=GetDocumentStatusResponse)
def get_document_status(doc_hash):
    logger.info("Processing get document status request...")
    try:
        doc_name = app.state.db_client.get_document_name(doc_hash)
        if not doc_name:
            raise HTTPException(status_code=404)
        status = app.state.db_client.get_document_status(doc_hash)
        return GetDocumentStatusResponse(doc_name=doc_name, status=status)
    except SQLAlchemyError:
        logger.error("DB operation failed")
        raise HTTPException(status_code=503)


@app.put("/documents/{doc_hash}/status/", response_model=DMSDocument | None)
def put_document_status(doc_hash, request: SetDocumentStatusRequest):
    logger.info("Processing put document status request...")
    try:
        document, result = app.state.db_client.set_document_status(
            doc_hash, request.doc_name, request.status
        )
    except DocumentHashConflictException:
        raise HTTPException(status_code=409)
    except (SQLAlchemyError, ValidationError):
        logger.error("DB operation failed")
        raise HTTPException(status_code=503)
    if result is SetDocumentResult.UPDATED:
        return Response(status_code=HTTP_204_NO_CONTENT)
    return Response(
        status_code=HTTP_201_CREATED,
        content=document.model_dump_json(),
        headers={"Content-Type": "application/json"},
    )


@app.get("/documents/", response_model=List[DMSDocument])
def get_documents():
    logger.info("Processing get documents request...")
    try:
        docs = app.state.db_client.get_documents()
        if not docs:
            raise HTTPException(status_code=204)
        return docs
    except (SQLAlchemyError, ValidationError):
        logger.error("DB operation failed")
        raise HTTPException(status_code=503)
