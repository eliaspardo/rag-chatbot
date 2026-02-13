import asyncio
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded dependencies (kept as module attributes so tests can patch them).
SessionManager = None
prepare_vector_store = None
FileLoader = None
get_rag_preprocessor = None
ExamPrepCore = None
ServerSetupException = None
FaissException = None
NoDocumentsException = None
VectorStoreException = None
Error = None


def _load_dependencies() -> None:
    global SessionManager
    global prepare_vector_store
    global FileLoader
    global get_rag_preprocessor
    global ExamPrepCore
    global ServerSetupException
    global FaissException
    global NoDocumentsException
    global VectorStoreException
    global Error

    if SessionManager is None:
        from src.api.session_manager import SessionManager as _SessionManager

        SessionManager = _SessionManager
    if prepare_vector_store is None:
        from src.core.app_bootstrap import prepare_vector_store as _prepare_vector_store

        prepare_vector_store = _prepare_vector_store
    if FileLoader is None:
        from src.core.file_loader import FileLoader as _FileLoader

        FileLoader = _FileLoader
    if get_rag_preprocessor is None:
        from src.core.rag_preprocessor import (
            get_rag_preprocessor as _get_rag_preprocessor,
        )

        get_rag_preprocessor = _get_rag_preprocessor
    if ExamPrepCore is None:
        from src.core.exam_prep_core import ExamPrepCore as _ExamPrepCore

        ExamPrepCore = _ExamPrepCore
    if (
        ServerSetupException is None
        or FaissException is None
        or NoDocumentsException is None
        or VectorStoreException is None
    ):
        from src.core.exceptions import (
            FaissException as _FaissException,
            NoDocumentsException as _NoDocumentsException,
            ServerSetupException as _ServerSetupException,
            VectorStoreException as _VectorStoreException,
        )

        ServerSetupException = _ServerSetupException
        FaissException = _FaissException
        NoDocumentsException = _NoDocumentsException
        VectorStoreException = _VectorStoreException
    if Error is None:
        from src.core.constants import Error as _Error

        Error = _Error


def initialize_services(app, progress_callback=print) -> None:
    # Keep heavy imports out of module import path so uvicorn can bind quickly.
    _load_dependencies()

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
