from fastapi import FastAPI
from src.document_management_service.lifespan import lifespan
from src.shared.models import GetDocumentStatusResponse
import logging

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/document/{doc_hash}/status/", response_model=GetDocumentStatusResponse)
def get_document_status(doc_hash):
    print("Processing ingestion request...")
    doc_name = app.state.db_client.get_document_name(doc_hash)
    status = app.state.db_client.get_document_status(doc_hash)
    return GetDocumentStatusResponse(doc_name=doc_name, status=status)
