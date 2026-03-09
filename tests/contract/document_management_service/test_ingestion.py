import threading
import time
from typing import Any
from unittest.mock import Mock
import uvicorn
import pytest
from pact import Verifier

from src.document_management_service.main import app
from src.shared.constants import DocumentStatus


# Module-level mock — visible to fixture and state handlers
mock_db_client = Mock()


def document_completed(
    parameters: dict[str, Any] | None,
) -> None:
    """Mock the provider state where the document is COMPLETED."""
    parameters = parameters or {}
    doc_hash = parameters.get("doc_hash", 123)

    # mock database to have document as COMPLETED
    mock_db_client.get_document_name(doc_hash).return_value = "Test Name"
    mock_db_client.get_document_status(doc_hash).return_value = DocumentStatus.COMPLETED
    return


def document_error(
    parameters: dict[str, Any] | None,
) -> None:
    """Mock the provider state where the document is ERROR."""
    parameters = parameters or {}
    doc_hash = parameters.get("doc_hash", 123)

    # mock database to have document as ERROR
    mock_db_client.get_document_name(doc_hash).return_value = "Test Name"
    mock_db_client.get_document_status(doc_hash).return_value = DocumentStatus.ERROR
    return


@pytest.fixture(scope="session")
def application():
    """Start up application for provider tests."""
    app.state.db_client = mock_db_client

    config = uvicorn.Config(app, host="0.0.0.0", port=8004)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1)  # wait for server to be ready
    yield
    server.should_exit = True
    thread.join(timeout=5)


class TestIngestion:
    # Map state names to handler functions
    state_handlers = {
        "document d41d8cd98f00b204e9800998ecf8427e is DocumentStatus.COMPLETED": document_completed,
        "document d41d8cd98f00b204e9800998ecf8427e is DocumentStatus.ERROR": document_error,
    }

    def test_provider_from_broker(self, application):
        """Test the provider against contracts from a Pact Broker."""
        verifier = (
            Verifier("my-provider")
            .add_transport(url="http://localhost:8004")
            .broker_source(
                "http://localhost:9292/",
            )
            .state_handler(self.state_handlers, teardown=True)
        )

        verifier.verify()
