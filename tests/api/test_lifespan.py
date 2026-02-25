import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.inference_service.api.lifespan import lifespan
from src.shared.exceptions import (
    ChromaException,
    NoDocumentsException,
    ServerSetupException,
    VectorStoreException,
)


def run_lifespan(app):
    async def _runner():
        async with lifespan(app):
            pass

    asyncio.run(_runner())


class TestLifespan:
    @patch("src.inference_service.api.lifespan.ExamPrepCore")
    @patch("src.inference_service.api.lifespan.SessionManager")
    @patch("src.inference_service.api.lifespan.prepare_vector_store")
    @patch("src.inference_service.api.lifespan.FileLoader")
    @patch("src.inference_service.api.lifespan.get_rag_preprocessor")
    def test_lifespan_success(
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

        run_lifespan(app)

        mock_prepare_vector_store.assert_called_once_with(
            rag_preprocessor=mock_get_rag_preprocessor.return_value,
            file_loader=mock_file_loader.return_value,
            progress_callback=print,
        )
        mock_session_manager.assert_called_once_with(vectordb)
        mock_exam_prep_core.assert_called_once_with(vectordb)

    @patch("src.inference_service.api.lifespan.FileLoader")
    def test_lifespan_file_loader_error(self, mock_file_loader):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_file_loader.side_effect = Exception("bad config")

        with pytest.raises(ServerSetupException):
            run_lifespan(app)

    @pytest.mark.parametrize(
        "exception",
        [
            NoDocumentsException(),
            ChromaException(),
            VectorStoreException(),
            Exception("unexpected"),
        ],
    )
    @patch("src.inference_service.api.lifespan.prepare_vector_store")
    @patch("src.inference_service.api.lifespan.FileLoader")
    @patch("src.inference_service.api.lifespan.get_rag_preprocessor")
    def test_lifespan_prepare_vector_store_errors(
        self,
        mock_get_rag_preprocessor,
        mock_file_loader,
        mock_prepare_vector_store,
        exception,
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_prepare_vector_store.side_effect = exception

        with pytest.raises(ServerSetupException):
            run_lifespan(app)

    @patch("src.inference_service.api.lifespan.SessionManager")
    @patch("src.inference_service.api.lifespan.prepare_vector_store")
    @patch("src.inference_service.api.lifespan.FileLoader")
    @patch("src.inference_service.api.lifespan.get_rag_preprocessor")
    def test_lifespan_session_manager_error(
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
            run_lifespan(app)

    @patch("src.inference_service.api.lifespan.ExamPrepCore")
    @patch("src.inference_service.api.lifespan.prepare_vector_store")
    @patch("src.inference_service.api.lifespan.FileLoader")
    @patch("src.inference_service.api.lifespan.get_rag_preprocessor")
    def test_lifespan_exam_prep_core_error(
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
            run_lifespan(app)
