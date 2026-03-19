import threading
import time
from typing import Any
from unittest.mock import Mock, patch

import pytest
import uvicorn
from pact import Verifier

from src.inference_service.main import app


mock_vector_store_loader = Mock()
mock_dms_client = Mock()


def given_has_documents_loaded(parameters: dict[str, Any] | None = None) -> None:
    mock_vector_store_loader.get_collection_count.return_value = 2
    mock_documents = [
        Mock(
            model_dump=lambda: {
                "doc_hash": "abc123",
                "doc_name": "doc1.pdf",
                "status": "Document processing completed",
            }
        ),
        Mock(
            model_dump=lambda: {
                "doc_hash": "def456",
                "doc_name": "doc2.pdf",
                "status": "Document pending processing",
            }
        ),
    ]
    mock_dms_client.get_documents.return_value = mock_documents


def given_has_no_documents(parameters: dict[str, Any] | None = None) -> None:
    mock_vector_store_loader.get_collection_count.return_value = 0
    mock_dms_client.get_documents.return_value = []


@pytest.fixture(scope="session")
def application():
    """Start up inference service for provider tests."""
    app.state.vector_store_loader = mock_vector_store_loader
    app.state.dms_client = mock_dms_client

    with patch("src.inference_service.main.app.state.vector_store_loader", mock_vector_store_loader):
        with patch("src.inference_service.main.app.state.dms_client", mock_dms_client):
            config = uvicorn.Config(app, host="0.0.0.0", port=8045)
            server = uvicorn.Server(config)
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            time.sleep(1)
            yield
            server.should_exit = True
            thread.join(timeout=5)


class TestUiService:
    state_handlers = {
        "Inference service has documents loaded": given_has_documents_loaded,
        "Inference service has no documents": given_has_no_documents,
    }

    def test_provider_from_broker(self, application):
        """Test the inference service provider against contracts from ui-service consumer."""
        verifier = (
            Verifier("inference-service")
            .add_transport(url="http://localhost:8045")
            .broker_source(
                "http://localhost:9292/",
            )
            .state_handler(self.state_handlers, teardown=True)
        )

        verifier.verify()
