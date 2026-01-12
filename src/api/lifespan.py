from contextlib import asynccontextmanager

from src.api.session_manager import SessionManager
from src.core.app_bootstrap import prepare_vector_store
from src.core.rag_preprocessor import RAGPreprocessor
from src.core.exceptions import (
    ServerSetupException,
    FaissException,
    NoDocumentsException,
    VectorStoreException,
)
from src.core.constants import Error
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # Startup
    print("Loading vector store...")
    rag_preprocessor = RAGPreprocessor()
    try:
        vectordb = prepare_vector_store(rag_preprocessor=rag_preprocessor)
    except NoDocumentsException:
        logger.error(Error.NO_DOCUMENTS)
        raise ServerSetupException()
    except (FaissException, VectorStoreException) as exception:
        logger.error(Error.EXCEPTION, exception)
        raise ServerSetupException()
    except Exception as exception:
        logger.error(Error.EXCEPTION, exception)
        raise ServerSetupException()
    print("Vector store loaded")

    try:
        app.state.session_manager: SessionManager = SessionManager(vectordb)
    except Exception as exception:
        logger.error(Error.EXCEPTION, exception)
        raise ServerSetupException()
    yield

    # Shutdown
    print("Cleaning up...")
