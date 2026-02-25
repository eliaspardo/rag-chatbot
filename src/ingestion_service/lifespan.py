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
    print("Processing vector store...")
    vector_store_builder = get_vector_store_builder()
    try:
        file_loader = FileLoader()
    except Exception:
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    try:
        app.state.vectordb = prepare_vector_store(
            vector_store_builder=vector_store_builder,
            file_loader=file_loader,
            progress_callback=print,
        )
    except NoDocumentsException:
        logger.error(Error.NO_DOCUMENTS)
        raise ServerSetupException()
    except (ChromaException, VectorStoreException):
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    except Exception:
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    print("Vector store ready!")

    yield

    # Shutdown
    print("Cleaning up...")
