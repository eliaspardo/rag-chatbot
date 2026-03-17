from contextlib import asynccontextmanager

from src.inference_service.session_manager import SessionManager
from src.inference_service.bootstrap import prepare_vector_store
from src.inference_service.core.vector_store_loader import get_vector_store_loader
from src.inference_service.core.exam_prep_core import ExamPrepCore
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
    print("Loading vector store...")
    app.state.vector_store_loader = get_vector_store_loader()
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
    print("Vector store loaded")

    try:
        app.state.session_manager: SessionManager = SessionManager(vectordb)
    except Exception:
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    try:
        app.state.exam_prep_core = ExamPrepCore(vectordb)
    except Exception:
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    yield

    # Shutdown
    print("Cleaning up...")
