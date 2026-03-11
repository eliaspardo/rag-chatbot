from typing import List
from fastapi import FastAPI, HTTPException, Response
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT
from src.document_management_service.lifespan import lifespan
from src.shared.constants import SetDocumentResult
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
    print("Processing get document status request...")
    doc_name = app.state.db_client.get_document_name(doc_hash)
    if not doc_name:
        raise HTTPException(status_code=404)
    status = app.state.db_client.get_document_status(doc_hash)
    print(f"Doc name: {doc_name}")
    print(f"Doc status: {status}")
    return GetDocumentStatusResponse(doc_name=doc_name, status=status)


@app.put("/documents/{doc_hash}/status/", response_model=DMSDocument | None)
def put_document_status(doc_hash, request: SetDocumentStatusRequest):
    print("Processing put document status request...")
    print(f"Doc name: {request.doc_name}")
    print(f"Doc status: {request.status}")
    document, result = app.state.db_client.set_document_status(
        doc_hash, request.doc_name, request.status
    )
    if result is SetDocumentResult.UPDATED:
        return Response(status_code=HTTP_204_NO_CONTENT)
    return Response(
        status_code=HTTP_201_CREATED,
        content=document.model_dump_json(),
        headers={"Content-Type": "application/json"},
    )


@app.get("/documents/", response_model=List[DMSDocument])
def get_documents():
    print("Processing get documents request...")
    docs = app.state.db_client.get_documents()
    if not docs:
        raise HTTPException(status_code=204)
    return docs
