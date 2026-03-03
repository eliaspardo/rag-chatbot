from typing import Generator
import pytest
from pact import Pact
from requests import HTTPError
from src.shared.constants import DocumentStatus


@pytest.fixture
def pact() -> Generator[Pact, None, None]:  #
    """Set up a Pact mock provider for consumer tests."""
    pact = Pact("ingestion-service", "document-management-service").with_specification(
        "V4"
    )
    yield pact
    pact.write_file("pacts")


def test_check_document_unknown_to_DMS(pact):
    doc_id = "abc-123"

    (
        pact.upon_receiving(f"A request to get {doc_id} status")
        .given(f"DMS has no knowledge of document {doc_id}")
        .with_request("GET", f"/documents/{doc_id}/status")
        .will_respond_with(404)
    )

    with pact.serve() as srv:
        from src.ingestion_service.document_management_client import get_document_status

        response = get_document_status(doc_id, base_url=srv.url)
        assert response is None


def test_check_document_DMS_internal_error(pact):
    doc_id = "abc-123"

    (
        pact.upon_receiving(f"A request to get {doc_id} status")
        .given("DMS is returning 503")
        .with_request("GET", f"/documents/{doc_id}/status")
        .will_respond_with(503)
    )

    with pact.serve() as srv:
        from src.ingestion_service.document_management_client import get_document_status

        with pytest.raises(HTTPError):
            get_document_status(doc_id, base_url=srv.url)


def test_check_document_is_being_processed(pact):
    doc_id = "abc-123"
    response = {
        "doc_id": doc_id,
        "doc_hash": "fasdfasdf",
        "status": DocumentStatus.PENDING,
    }

    (
        pact.upon_receiving(f"A request to get {doc_id} status")
        .given(f"Document {doc_id} is being processed")
        .with_request("GET", f"/documents/{doc_id}/status")
        .will_respond_with(200)
        .with_body(response)
    )

    with pact.serve() as srv:
        from src.ingestion_service.document_management_client import get_document_status

        response = get_document_status(doc_id, base_url=srv.url)
        assert response == DocumentStatus.PENDING
