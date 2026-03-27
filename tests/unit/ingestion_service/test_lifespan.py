import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.ingestion_service.lifespan import lifespan
from src.shared.exceptions import ConfigurationException, ServerSetupException


def run_lifespan(app):
    async def _runner():
        async with lifespan(app):
            pass

    asyncio.run(_runner())


class TestLifespan:
    @patch("src.ingestion_service.lifespan.get_vector_store_builder")
    def test_lifespan_success(
        self,
        mock_get_vector_store_builder,
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_vector_store_builder = Mock()
        mock_get_vector_store_builder.return_value = mock_vector_store_builder

        run_lifespan(app)

        mock_get_vector_store_builder.assert_called_once()

    @patch("src.ingestion_service.lifespan.get_vector_store_builder")
    @patch("src.ingestion_service.lifespan.FileLoader")
    def test_lifespan_file_loader_error(
        self, mock_file_loader, mock_get_vector_store_builder
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_vector_store_builder = Mock()
        mock_get_vector_store_builder.return_value = mock_vector_store_builder
        mock_file_loader.side_effect = ConfigurationException()

        with pytest.raises(ServerSetupException):
            run_lifespan(app)

    @patch.dict("os.environ", {"DMS_URL": ""}, clear=False)
    @patch("src.ingestion_service.lifespan.get_vector_store_builder")
    def test_lifespan_no_dms_URL_error(
        self,
        mock_get_vector_store_builder,
    ):
        app = SimpleNamespace(state=SimpleNamespace())
        mock_vector_store_builder = Mock()
        mock_get_vector_store_builder.return_value = mock_vector_store_builder

        with pytest.raises(ServerSetupException):
            run_lifespan(app)
