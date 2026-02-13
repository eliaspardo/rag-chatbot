import asyncio
from contextlib import asynccontextmanager
import logging

from src.api.session_manager import SessionManager
from src.core.app_bootstrap import prepare_vector_store
from src.core.file_loader import FileLoader
from src.core.rag_preprocessor import get_rag_preprocessor
from src.core.exam_prep_core import ExamPrepCore
from src.core.exceptions import (
    ServerSetupException,
    FaissException,
    NoDocumentsException,
    VectorStoreException,
)
from src.core.constants import Error

logger = logging.getLogger(__name__)


def initialize_services(app, progress_callback=print) -> None:
    rag_preprocessor = get_rag_preprocessor()
    try:
        file_loader = FileLoader()
    except Exception:
        logger.error(Error.EXCEPTION, exc_info=True)
        raise ServerSetupException()
    try:
        vectordb = prepare_vector_store(
            rag_preprocessor=rag_preprocessor,
            file_loader=file_loader,
            progress_callback=progress_callback,
        )
    except NoDocumentsException:
        logger.error(Error.NO_DOCUMENTS)
        raise ServerSetupException()
    except (FaissException, VectorStoreException):
        logger.error(Error.EXCEPTION, exc_info=True)
        raise ServerSetupException()
    except Exception:
        logger.error(Error.EXCEPTION, exc_info=True)
        raise ServerSetupException()

    try:
        app.state.session_manager = SessionManager(vectordb)
    except Exception:
        logger.error(Error.EXCEPTION, exc_info=True)
        raise ServerSetupException()
    try:
        app.state.exam_prep_core = ExamPrepCore(vectordb)
    except Exception:
        logger.error(Error.EXCEPTION, exc_info=True)
        raise ServerSetupException()


async def _bootstrap_services(app) -> None:
    try:
        logger.info("Bootstrapping application services...")
        await asyncio.to_thread(initialize_services, app, print)
        app.state.bootstrap_ready = True
        logger.info("Application services initialized.")
    except Exception as exception:
        app.state.bootstrap_error = str(exception)
        logger.exception("Service bootstrap failed")


@asynccontextmanager
async def lifespan(app):
    app.state.bootstrap_ready = False
    app.state.bootstrap_error = None
    app.state.bootstrap_task = asyncio.create_task(_bootstrap_services(app))
    yield

    # Shutdown
    print("Cleaning up...")
    bootstrap_task = getattr(app.state, "bootstrap_task", None)
    if bootstrap_task and not bootstrap_task.done():
        bootstrap_task.cancel()
