import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.inference_service.lifespan import lifespan
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
    @patch("src.inference_service.lifespan.SessionManager")
    @patch("src.inference_service.lifespan.prepare_vector_store")
    @patch("src.inference_service.lifespan.get_vector_store_loader")
    def test_lifespan_success(
        self,
        mock_get_vector_store_loader,
        mock_prepare_vector_store,
        mock_session_manager,
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        vectordb = Mock()
        mock_prepare_vector_store.return_value = vectordb

        run_lifespan(app)

        mock_prepare_vector_store.assert_called_once_with(
            vector_store_loader=mock_get_vector_store_loader.return_value,
            progress_callback=print,
        )
        mock_session_manager.assert_called_once_with(vectordb)

    @pytest.mark.parametrize(
        "exception",
        [
            NoDocumentsException(),
            ChromaException(),
            VectorStoreException(),
            Exception("unexpected"),
        ],
    )
    @patch("src.inference_service.lifespan.prepare_vector_store")
    @patch("src.inference_service.lifespan.get_vector_store_loader")
    def test_lifespan_prepare_vector_store_errors(
        self,
        mock_get_vector_store_loader,
        mock_prepare_vector_store,
        exception,
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_prepare_vector_store.side_effect = exception

        with pytest.raises(ServerSetupException):
            run_lifespan(app)

    @patch("src.inference_service.lifespan.prepare_vector_store")
    @patch("src.inference_service.lifespan.get_vector_store_loader")
    @patch(
        "src.inference_service.lifespan.DMS_URL",
        "",
    )
    def test_lifespan_dms_env_var_error(
        self,
        mock_get_vector_store_loader,
        mock_prepare_vector_store,
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_prepare_vector_store.return_value = Mock()

        with pytest.raises(
            ServerSetupException, match="DMS_URL environment variable is required"
        ):
            run_lifespan(app)

    @patch("src.inference_service.lifespan.SessionManager")
    @patch("src.inference_service.lifespan.prepare_vector_store")
    @patch("src.inference_service.lifespan.get_vector_store_loader")
    def test_lifespan_session_manager_error(
        self,
        mock_get_vector_store_loader,
        mock_prepare_vector_store,
        mock_session_manager,
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_prepare_vector_store.return_value = Mock()
        mock_session_manager.side_effect = Exception("session error")

        with pytest.raises(ServerSetupException):
            run_lifespan(app)
