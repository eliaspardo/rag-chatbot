from contextlib import asynccontextmanager

from src.ingestion_service.bootstrap import prepare_vector_store
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.rag_preprocessor import get_rag_preprocessor
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
    rag_preprocessor = get_rag_preprocessor()
    try:
        file_loader = FileLoader()
    except Exception:
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    try:
        app.state.vectordb = prepare_vector_store(
            rag_preprocessor=rag_preprocessor,
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
