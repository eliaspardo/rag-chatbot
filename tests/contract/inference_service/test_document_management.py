from typing import Generator
import pytest
from pact import Pact

from src.inference_service.document_management_client import DocumentManagementClient
from src.shared.constants import DocumentStatus
from src.shared.models import DMSDocument


@pytest.fixture
def pact() -> Generator[Pact, None, None]:  #
    """Set up a Pact mock provider for consumer tests."""
    pact = Pact("inference-service", "document-management-service").with_specification(
        "V4"
    )
    yield pact
    pact.write_file("pacts")


class TestDocumentManagement:
    def test_get_documents_empty(self, pact):
        (
            pact.upon_receiving("Get documents when DMS has no documents")
            .given("DMS has no documents")
            .with_request("GET", "/documents/")
            .will_respond_with(204)
        )

        with pact.serve() as srv:
            dms_client = DocumentManagementClient(srv.url)
            result = dms_client.get_documents()
            assert result == []

    def test_get_documents_one_doc(self, pact):
        response = [
            {
                "doc_hash": "Doc Hash 1",
                "doc_name": "Doc Name 1",
                "status": DocumentStatus.PENDING,
            },
        ]
        (
            pact.upon_receiving("Get documents when DMS has one document")
            .given("DMS has one document")
            .with_request("GET", "/documents/")
            .will_respond_with(200)
            .with_body(response)
        )

        with pact.serve() as srv:
            dms_client = DocumentManagementClient(srv.url)
            result = dms_client.get_documents()
            assert result == [DMSDocument(**item) for item in response]

    def test_get_documents_three_docs(self, pact):
        response = [
            {
                "doc_hash": "Doc Hash 1",
                "doc_name": "Doc Name 1",
                "status": DocumentStatus.PENDING,
            },
            {
                "doc_hash": "Doc Hash 2",
                "doc_name": "Doc Name 2",
                "status": DocumentStatus.COMPLETED,
            },
            {
                "doc_hash": "Doc Hash 3",
                "doc_name": "Doc Name 3",
                "status": DocumentStatus.ERROR,
            },
        ]
        (
            pact.upon_receiving("Get documents when DMS has multiple documents")
            .given("DMS has multiple documents")
            .with_request("GET", "/documents/")
            .will_respond_with(200)
            .with_body(response)
        )

        with pact.serve() as srv:
            dms_client = DocumentManagementClient(srv.url)
            result = dms_client.get_documents()
            assert result == [DMSDocument(**item) for item in response]
