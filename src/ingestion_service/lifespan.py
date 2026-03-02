from contextlib import asynccontextmanager

from src.ingestion_service.bootstrap import prepare_vector_store
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import get_vector_store_builder
from src.shared.exceptions import (
    ChromaException,
    ServerSetupException,
    NoDocumentsException,
    VectorStoreException,
)
from src.shared.constants import Error
import logging

logger = logging.getLogger(__name__)


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
    try:
        prepare_vector_store(
            vector_store_builder=app.state.vector_store_builder,
            file_loader=app.state.file_loader,
            progress_callback=print,
        )
        print("Vector store ready!")
    except NoDocumentsException:
        logger.error("Running with empty vector store. Make sure to ingest documents.")
    except (ChromaException, VectorStoreException):
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    except Exception:
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()

    yield

    # Shutdown
    print("Cleaning up...")
