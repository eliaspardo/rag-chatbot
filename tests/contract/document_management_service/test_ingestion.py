from functools import partial
import threading
import time
from typing import Any
from unittest.mock import Mock, patch
import uvicorn
import pytest
from pact import Verifier

from src.document_management_service.db_client import DBClient
from src.document_management_service.main import app, get_db_client
from src.shared.constants import DocumentStatus, SetDocumentResult
from src.shared.exceptions import DocumentHashConflictException
from src.shared.models import DMSDocument

sample_hash = "d41d8cd98f00b204e9800998ecf8427e"
sample_doc_name = "Test doc name"

mock_db_client = Mock(spec=DBClient)


def given_document_has_status(
    parameters: dict[str, Any] | None, doc_name: str, status: DocumentStatus
) -> None:
    # mock database to have document in status
    mock_db_client.get_document_name.return_value = doc_name
    mock_db_client.get_document_status.return_value = status

    return


def given_document_not_found(parameters: dict[str, Any] | None) -> None:
    # mock database to have not have document
    mock_db_client.get_document_name.return_value = None
    return


def given_dms_has_no_documents() -> None:
    # mock database to have not have documents
    mock_db_client.get_documents.return_value = None
    return


def given_dms_has_two_documents() -> None:
    document_1 = DMSDocument(
        doc_hash=sample_hash, doc_name=sample_doc_name, status=DocumentStatus.PENDING
    )
    document_2 = DMSDocument(
        doc_hash="Doc Hash 2", doc_name="Doc Name 2", status=DocumentStatus.COMPLETED
    )
    mock_db_client.get_documents.return_value = [document_1, document_2]


def given_dms_has_one_document() -> None:
    document = DMSDocument(
        doc_hash="Doc Hash 1", doc_name="Doc Name 1", status=DocumentStatus.PENDING
    )
    mock_db_client.get_documents.return_value = [document]


def given_dms_has_multiple_documents() -> None:
    document_1 = DMSDocument(
        doc_hash="Doc Hash 1", doc_name="Doc Name 1", status=DocumentStatus.PENDING
    )
    document_2 = DMSDocument(
        doc_hash="Doc Hash 2", doc_name="Doc Name 2", status=DocumentStatus.COMPLETED
    )
    document_3 = DMSDocument(
        doc_hash="Doc Hash 3", doc_name="Doc Name 3", status=DocumentStatus.ERROR
    )
    mock_db_client.get_documents.return_value = [document_1, document_2, document_3]


def given_document_exists_with_name() -> None:
    mock_db_client.set_document_status.side_effect = DocumentHashConflictException()


def given_document_exists() -> None:
    result = SetDocumentResult.UPDATED
    mock_db_client.set_document_status.side_effect = (
        lambda doc_hash, doc_name, status: (
            DMSDocument(doc_hash=doc_hash, doc_name=doc_name, status=status),
            result,
        )
    )


def given_document_does_not_exist() -> None:
    result = SetDocumentResult.CREATED
    mock_db_client.set_document_status.side_effect = (
        lambda doc_hash, doc_name, status: (
            DMSDocument(doc_hash=doc_hash, doc_name=doc_name, status=status),
            result,
        )
    )


@pytest.fixture(scope="session")
def application():
    """Start up application for provider tests."""
    # Patch the module-level variable that's already been read at import time
    with patch(
        "src.document_management_service.lifespan.DMS_DATABASE_URL",
        "sqlite:///:memory:",
    ):
        # Use FastAPI's dependency override for generator dependencies
        app.dependency_overrides[get_db_client] = lambda: mock_db_client

        config = uvicorn.Config(app, host="0.0.0.0", port=8044)
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        time.sleep(1)  # wait for server to be ready
        yield
        server.should_exit = True
        thread.join(timeout=5)
        app.dependency_overrides.clear()


class TestIngestion:
    # Map state names to handler functions
    state_handlers = {
        f"Document {sample_hash} is DocumentStatus.COMPLETED": partial(
            given_document_has_status,
            doc_name=sample_doc_name,
            status=DocumentStatus.COMPLETED,
        ),
        f"Document {sample_hash} is DocumentStatus.ERROR": partial(
            given_document_has_status,
            doc_name=sample_doc_name,
            status=DocumentStatus.ERROR,
        ),
        f"Document {sample_hash} is DocumentStatus.PENDING": partial(
            given_document_has_status,
            doc_name=sample_doc_name,
            status=DocumentStatus.PENDING,
        ),
        f"DMS has no knowledge of document {sample_hash}": given_document_not_found,
        "DMS has no documents registered": given_dms_has_no_documents,
        "DMS has documents registered": given_dms_has_two_documents,
        "DMS has multiple documents": given_dms_has_multiple_documents,
        f"Document {sample_hash} exists in the db with doc_name {sample_doc_name}": given_document_exists_with_name,
        f"Document {sample_hash} already exists in the db": given_document_exists,
        f"Document {sample_hash} does not exist in the db": given_document_does_not_exist,
    }

    def test_provider_from_broker(self, application):
        """Test the provider against contracts from a Pact Broker."""
        verifier = (
            Verifier("document-management-service")
            .add_transport(url="http://localhost:8044")
            .broker_source(
                "http://localhost:9292/",
            )
            .state_handler(self.state_handlers, teardown=True)
        )

        verifier.verify()
