"""Lifespan context manager for the inference service FastAPI application."""

from contextlib import asynccontextmanager
import os

from src.inference_service.document_management_client import DocumentManagementClient
from src.inference_service.session_manager import SessionManager
from src.inference_service.bootstrap import prepare_vector_store
from src.inference_service.core.vector_store_loader import get_vector_store_loader
from src.shared.env_loader import load_environment
from src.shared.exceptions import (
    ChromaException,
    ServerSetupException,
    NoDocumentsException,
    VectorStoreException,
)
from src.shared.constants import Error
import logging
import mlflow
import mlflow.langchain

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    """Initialize and tear down inference service resources on application startup/shutdown."""
    # Startup
    logger.info("Setting up MLflow autologging...")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("inference-service")
    mlflow.langchain.autolog(log_traces=True)
    logger.info("Loading vector store...")
    app.state.vector_store_loader = get_vector_store_loader()
    load_environment()
    DMS_URL = os.getenv("DMS_URL")
    if not DMS_URL:
        raise ServerSetupException("DMS_URL environment variable is required")
    app.state.dms_client = DocumentManagementClient(DMS_URL)
    try:
        vectordb = prepare_vector_store(
            vector_store_loader=app.state.vector_store_loader,
            progress_callback=print,
        )
        app.state.vectordb = vectordb
    except NoDocumentsException:
        logger.error(Error.NO_DOCUMENTS)
        raise ServerSetupException()
    except (ChromaException, VectorStoreException):
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    except Exception:
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    logger.info("Vector store loaded")

    try:
        app.state.session_manager: SessionManager = SessionManager(vectordb)
    except Exception:
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    yield

    # Shutdown
    logger.info("Cleaning up...")
    mlflow.end_run()
