import asyncio
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import pytest

from src.api.lifespan import initialize_services, lifespan
from src.core.exceptions import (
    FaissException,
    NoDocumentsException,
    ServerSetupException,
    VectorStoreException,
)


def run_lifespan_once(app):
    async def _run():
        async with lifespan(app):
            await asyncio.sleep(0)

    asyncio.run(_run())


class TestLifespan:
    @patch("src.api.lifespan.ExamPrepCore")
    @patch("src.api.lifespan.SessionManager")
    @patch("src.api.lifespan.prepare_vector_store")
    @patch("src.api.lifespan.FileLoader")
    @patch("src.api.lifespan.get_rag_preprocessor")
    def test_initialize_services_success(
        self,
        mock_get_rag_preprocessor,
        mock_file_loader,
        mock_prepare_vector_store,
        mock_session_manager,
        mock_exam_prep_core,
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        vectordb = Mock()
        mock_prepare_vector_store.return_value = vectordb

        initialize_services(app, progress_callback=Mock())

        mock_prepare_vector_store.assert_called_once_with(
            rag_preprocessor=mock_get_rag_preprocessor.return_value,
            file_loader=mock_file_loader.return_value,
            progress_callback=ANY,
        )
        mock_session_manager.assert_called_once_with(vectordb)
        mock_exam_prep_core.assert_called_once_with(vectordb)

    @patch("src.api.lifespan.FileLoader")
    def test_initialize_services_file_loader_error(self, mock_file_loader):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_file_loader.side_effect = Exception("bad config")

        with pytest.raises(ServerSetupException):
            initialize_services(app)

    @pytest.mark.parametrize(
        "exception",
        [
            NoDocumentsException(),
            FaissException(),
            VectorStoreException(),
            Exception("unexpected"),
        ],
    )
    @patch("src.api.lifespan.prepare_vector_store")
    @patch("src.api.lifespan.FileLoader")
    @patch("src.api.lifespan.get_rag_preprocessor")
    def test_initialize_services_prepare_vector_store_errors(
        self,
        mock_get_rag_preprocessor,
        mock_file_loader,
        mock_prepare_vector_store,
        exception,
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_prepare_vector_store.side_effect = exception

        with pytest.raises(ServerSetupException):
            initialize_services(app)

    @patch("src.api.lifespan.SessionManager")
    @patch("src.api.lifespan.prepare_vector_store")
    @patch("src.api.lifespan.FileLoader")
    @patch("src.api.lifespan.get_rag_preprocessor")
    def test_initialize_services_session_manager_error(
        self,
        mock_get_rag_preprocessor,
        mock_file_loader,
        mock_prepare_vector_store,
        mock_session_manager,
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_prepare_vector_store.return_value = Mock()
        mock_session_manager.side_effect = Exception("session error")

        with pytest.raises(ServerSetupException):
            initialize_services(app)

    @patch("src.api.lifespan.ExamPrepCore")
    @patch("src.api.lifespan.prepare_vector_store")
    @patch("src.api.lifespan.FileLoader")
    @patch("src.api.lifespan.get_rag_preprocessor")
    def test_initialize_services_exam_prep_core_error(
        self,
        mock_get_rag_preprocessor,
        mock_file_loader,
        mock_prepare_vector_store,
        mock_exam_prep_core,
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_prepare_vector_store.return_value = Mock()
        mock_exam_prep_core.side_effect = Exception("exam prep error")

        with pytest.raises(ServerSetupException):
            initialize_services(app)

    @patch("src.api.lifespan.asyncio.create_task")
    def test_lifespan_starts_background_bootstrap(self, mock_create_task):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_task = Mock()
        mock_task.done.return_value = False

        def _fake_create_task(coroutine):
            coroutine.close()
            return mock_task

        mock_create_task.side_effect = _fake_create_task

        run_lifespan_once(app)

        mock_create_task.assert_called_once()
        assert app.state.bootstrap_ready is False
        assert app.state.bootstrap_error is None
        mock_task.cancel.assert_called_once()
