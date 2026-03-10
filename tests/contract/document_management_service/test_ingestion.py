from functools import partial
import threading
import time
from typing import Any
from unittest.mock import Mock, patch
import uvicorn
import pytest
from pact import Verifier

from src.document_management_service.db_client import DBClient
from src.document_management_service.main import app
from src.shared.constants import DocumentStatus

sample_hash = "d41d8cd98f00b204e9800998ecf8427e"
sample_doc_name = "Test doc name"

mock_db_client = Mock(spec=DBClient)


def get_document_status(
    parameters: dict[str, Any] | None, doc_name: str, status: DocumentStatus
) -> None:
    # mock database to have document in status
    mock_db_client.get_document_name.return_value = doc_name
    mock_db_client.get_document_status.return_value = status

    return


def document_not_found(parameters: dict[str, Any] | None) -> None:
    # mock database to have not have document
    mock_db_client.get_document_name.return_value = None
    return


@pytest.fixture(scope="session")
def application():
    """Start up application for provider tests."""
    patcher = patch("src.document_management_service.lifespan.DBClient")
    mock_db_client_class = patcher.start()
    mock_db_client_class.return_value = mock_db_client

    config = uvicorn.Config(app, host="0.0.0.0", port=8004)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1)  # wait for server to be ready
    yield
    server.should_exit = True
    thread.join(timeout=5)
    patcher.stop()


class TestIngestion:
    # Map state names to handler functions
    state_handlers = {
        f"Document {sample_hash} is DocumentStatus.COMPLETED": partial(
            get_document_status,
            doc_name=sample_doc_name,
            status=DocumentStatus.COMPLETED,
        ),
        f"Document {sample_hash} is DocumentStatus.ERROR": partial(
            get_document_status, doc_name=sample_doc_name, status=DocumentStatus.ERROR
        ),
        f"Document {sample_hash} is DocumentStatus.PENDING": partial(
            get_document_status, doc_name=sample_doc_name, status=DocumentStatus.PENDING
        ),
        f"DMS has no knowledge of document {sample_hash}": document_not_found,
    }

    def test_provider_from_broker(self, application):
        """Test the provider against contracts from a Pact Broker."""
        verifier = (
            Verifier("document-management-service")
            .add_transport(url="http://localhost:8004")
            .broker_source(
                "http://localhost:9292/",
            )
            .state_handler(self.state_handlers, teardown=True)
        )

        verifier.verify()
