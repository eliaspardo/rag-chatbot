from typing import List
from fastapi import FastAPI, HTTPException
from src.document_management_service.lifespan import lifespan
from src.shared.models import DMSDocument, GetDocumentStatusResponse
import logging

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents/{doc_hash}/status", response_model=GetDocumentStatusResponse)
def get_document_status(doc_hash):
    print("Processing get document status request...")
    doc_name = app.state.db_client.get_document_name(doc_hash)
    if not doc_name:
        raise HTTPException(status_code=404)
    status = app.state.db_client.get_document_status(doc_hash)
    print(f"Doc name: {doc_name}")
    print(f"Doc status: {status}")
    return GetDocumentStatusResponse(doc_name=doc_name, status=status)


@app.put("/documents/{doc_hash}/status", response_model=GetDocumentStatusResponse)
def put_document_status(doc_hash):
    print("Processing put document status request...")
    # Does document exist?
    doc_name = app.state.db_client.get_document_name(doc_hash)
    status = app.state.db_client.get_document_status(doc_hash)
    print(f"Doc name: {doc_name}")
    print(f"Doc status: {status}")
    return GetDocumentStatusResponse(doc_name=doc_name, status=status)


@app.get("/documents", response_model=List[DMSDocument])
def get_documents():
    print("Processing get documents request...")
    docs = app.state.db_client.get_documents()
    if not docs:
        raise HTTPException(status_code=204)
    return docs
