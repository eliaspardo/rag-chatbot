from contextlib import asynccontextmanager
import os

from src.ingestion_service.document_ingestor import DocumentIngestor
from src.ingestion_service.document_management_client import DocumentManagementClient
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import get_vector_store_builder
from src.shared.env_loader import load_environment
from src.shared.exceptions import (
    ServerSetupException,
)
from src.shared.constants import Error
import logging

logger = logging.getLogger(__name__)

load_environment()
DMS_URL = os.getenv("DMS_URL")
PDF_PATH = os.getenv("PDF_PATH")


@asynccontextmanager
async def lifespan(app):
    # Startup
    print("Preparing vector store...")
    app.state.vector_store_builder = get_vector_store_builder()
    try:
        app.state.file_loader = FileLoader()
    except Exception:
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    print("Using DMS-enabled ingestion...")
    if not DMS_URL:
        raise ServerSetupException("DMS_URL environment variable is required")
    dms_client = DocumentManagementClient(DMS_URL)
    app.state.doc_ingestor = DocumentIngestor(
        dms_client,
        app.state.vector_store_builder,
        app.state.file_loader,
        print,
    )
    pdf_paths = (PDF_PATH or "").split(",")
    if PDF_PATH:
        app.state.doc_ingestor.ingest_documents(pdf_paths)
    print("Vector store ready!")

    yield

    # Shutdown
    print("Cleaning up...")
