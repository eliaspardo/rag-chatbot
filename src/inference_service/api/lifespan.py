from contextlib import asynccontextmanager

from src.inference_service.api.session_manager import SessionManager
from src.core.app_bootstrap import prepare_vector_store
from src.core.file_loader import FileLoader
from src.core.rag_preprocessor import get_rag_preprocessor
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
    rag_preprocessor = get_rag_preprocessor()
    try:
        file_loader = FileLoader()
    except Exception:
        logger.error(Error.EXCEPTION)
        raise ServerSetupException()
    try:
        vectordb = prepare_vector_store(
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
