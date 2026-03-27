import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.ingestion_service.lifespan import lifespan
from src.shared.exceptions import ConfigurationException, ServerSetupException


def run_lifespan(app):
    """
    Run the application's lifespan context manager to completion using a temporary event loop.
    
    Parameters:
        app: An application object compatible with `lifespan(app)` (e.g., has a `state` attribute). Any exception raised while entering or exiting the lifespan context is propagated to the caller.
    """
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
        """
        Verify that the lifespan context calls the vector store builder during startup.
        
        This test constructs a minimal app state, configures the patched `get_vector_store_builder`
        to return a mock builder, runs the lifespan, and asserts that `get_vector_store_builder`
        was invoked exactly once.
        
        Parameters:
            mock_get_vector_store_builder (Mock): Patch fixture for `src.ingestion_service.lifespan.get_vector_store_builder`.
        """
        app = SimpleNamespace(state=SimpleNamespace())
        mock_vector_store_builder = Mock()
        mock_get_vector_store_builder.return_value = mock_vector_store_builder

        run_lifespan(app)

        mock_get_vector_store_builder.assert_called_once()

    @patch("src.ingestion_service.lifespan.FileLoader")
    def test_lifespan_file_loader_error(
        self,
        mock_file_loader,
    ):
        """
        Verifies that a FileLoader configuration failure during application lifespan startup is surfaced as a ServerSetupException.
        
        Sets FileLoader to raise ConfigurationException and asserts that running the application's lifespan re-raises the error as ServerSetupException.
        """
        app = SimpleNamespace(state=SimpleNamespace())
        mock_file_loader.side_effect = ConfigurationException()

        with pytest.raises(ServerSetupException):
            run_lifespan(app)
