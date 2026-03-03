from typing import Generator
import pytest
from pact import Pact
from requests import HTTPError
from src.shared.constants import DocumentStatus

sample_hash = "d41d8cd98f00b204e9800998ecf8427e"


@pytest.fixture
def pact() -> Generator[Pact, None, None]:  #
    """Set up a Pact mock provider for consumer tests."""
    pact = Pact("ingestion-service", "document-management-service").with_specification(
        "V4"
    )
    yield pact
    pact.write_file("pacts")


def test_get_document_status_document_unknown_to_DMS(pact):
    (
        pact.upon_receiving("Get status for unknown document")
        .given(f"DMS has no knowledge of document {sample_hash}")
        .with_request("GET", f"/documents/{sample_hash}/status")
        .will_respond_with(404)
    )

    with pact.serve() as srv:
        from src.ingestion_service.document_management_client import get_document_status

        response = get_document_status(sample_hash, base_url=srv.url)
        assert response is None


def test_get_document_status_DMS_internal_error(pact):
    (
        pact.upon_receiving("Get status when DMS returns 503")
        .given("DMS is returning 503")
        .with_request("GET", f"/documents/{sample_hash}/status")
        .will_respond_with(503)
    )

    with pact.serve() as srv:
        from src.ingestion_service.document_management_client import get_document_status

        with pytest.raises(HTTPError) as exc_info:
            get_document_status(sample_hash, base_url=srv.url)
        assert exc_info.value.response.status_code == 503


def test_get_document_status_document_pending(pact):
    response = {
        "doc_name": "test_doc_name",
        "status": DocumentStatus.PENDING,
    }

    (
        pact.upon_receiving("Get status for pending document")
        .given(f"Document {sample_hash} is PENDING")
        .with_request("GET", f"/documents/{sample_hash}/status")
        .will_respond_with(200)
        .with_body(response)
    )

    with pact.serve() as srv:
        from src.ingestion_service.document_management_client import get_document_status

        response = get_document_status(sample_hash, base_url=srv.url)
        assert response == DocumentStatus.PENDING


def test_get_document_status_document_is_completed(pact):
    response = {
        "doc_name": "test_doc_name",
        "status": DocumentStatus.COMPLETED,
    }

    (
        pact.upon_receiving("Get status for completed document")
        .given(f"Document {sample_hash} is COMPLETED")
        .with_request("GET", f"/documents/{sample_hash}/status")
        .will_respond_with(200)
        .with_body(response)
    )

    with pact.serve() as srv:
        from src.ingestion_service.document_management_client import get_document_status

        response = get_document_status(sample_hash, base_url=srv.url)
        assert response == DocumentStatus.COMPLETED


def test_register_document_success(pact):
    doc_name = "test_doc_name"
    request = {"doc_name": doc_name}

    response = {
        "hash": sample_hash,
        "doc_name": doc_name,
        "status": DocumentStatus.INITIALIZED,
    }

    (
        pact.upon_receiving("Request to register new document")
        .given(f"Document {sample_hash} is not registered")
        .with_request("POST", f"/documents/{sample_hash}", body=request)
        .will_respond_with(201)
        .with_body(response)
    )

    with pact.serve() as srv:
        from src.ingestion_service.document_management_client import register_document

        response = register_document(sample_hash, doc_name, srv.url)
        assert response.doc_hash == sample_hash
        assert response.doc_name == doc_name
        assert response.status == DocumentStatus.INITIALIZED


def test_register_document_exists_error(pact):
    doc_name = "test_doc_name"
    request = {"doc_name": doc_name}

    (
        pact.upon_receiving("Request to register already existing document")
        .given(f"Document {sample_hash} is already registered")
        .with_request("POST", f"/documents/{sample_hash}", body=request)
        .will_respond_with(409)
        .with_body({"error": "Document already exists"})
    )

    with pact.serve() as srv:
        from src.ingestion_service.document_management_client import register_document

        with pytest.raises(HTTPError) as exc_info:
            register_document(sample_hash, doc_name, srv.url)
        assert exc_info.value.response.status_code == 409
        assert exc_info.value.response.json()["error"] == "Document already exists"


def test_register_document_malformed_error(pact):
    doc_name = ""
    request = {"doc_name": doc_name}

    (
        pact.upon_receiving("Request to register a document that is malformed")
        .with_request("POST", f"/documents/{sample_hash}", body=request)
        .will_respond_with(422)
        .with_body({"error": "Malformed request"})
    )

    with pact.serve() as srv:
        from src.ingestion_service.document_management_client import register_document

        with pytest.raises(HTTPError) as exc_info:
            register_document(sample_hash, doc_name, srv.url)
        assert exc_info.value.response.status_code == 422
        assert exc_info.value.response.json()["error"] == "Malformed request"
