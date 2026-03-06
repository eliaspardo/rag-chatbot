from contextlib import asynccontextmanager
import os

from src.ingestion_service.bootstrap import prepare_vector_store
from src.ingestion_service.document_ingestor import DocumentIngestor
from src.ingestion_service.document_management_client import DocumentManagementClient
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import get_vector_store_builder
from src.shared.env_loader import load_environment
from src.shared.exceptions import (
    ChromaException,
    ServerSetupException,
    NoDocumentsException,
    VectorStoreException,
)
from src.shared.constants import Error
import logging

logger = logging.getLogger(__name__)

load_environment()
DMS_ENABLED = os.getenv("DMS_ENABLED").lower() == "true"
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
    if DMS_ENABLED:
        print("Using DMS-enabled ingestion...")
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
    else:
        print("Using legacy ingestion...")
        try:
            prepare_vector_store(
                vector_store_builder=app.state.vector_store_builder,
                file_loader=app.state.file_loader,
                progress_callback=print,
            )
            print("Vector store ready!")
        except NoDocumentsException:
            logger.error(
                "Running with empty vector store. Make sure to ingest documents."
            )
        except (ChromaException, VectorStoreException):
            logger.error(Error.EXCEPTION)
            raise ServerSetupException()
        except Exception:
            logger.error(Error.EXCEPTION)
            raise ServerSetupException()

    yield

    # Shutdown
    print("Cleaning up...")
