from typing import Generator
import pytest
from pact import Pact
from src.ingestion_service.document_management_client import DocumentManagementClient
from src.shared.constants import DocumentStatus
from src.shared.models import DMSDocument

sample_hash = "d41d8cd98f00b204e9800998ecf8427e"
sample_doc_name = "Test doc name"


@pytest.fixture
def pact() -> Generator[Pact, None, None]:  #
    """Set up a Pact mock provider for consumer tests."""
    pact = Pact("ingestion-service", "document-management-service").with_specification(
        "V4"
    )
    yield pact
    pact.write_file("pacts")


def test_get_document_status_document_returns_404(pact):
    (
        pact.upon_receiving("Get status for unknown document")
        .given(f"DMS has no knowledge of document {sample_hash}")
        .with_request("GET", f"/documents/{sample_hash}/status")
        .will_respond_with(404)
    )

    with pact.serve() as srv:
        dms_client = DocumentManagementClient(srv.url)
        response = dms_client.get_document_status(sample_hash)
        assert response is None


@pytest.mark.parametrize(
    "status",
    [DocumentStatus.PENDING, DocumentStatus.COMPLETED, DocumentStatus.ERROR],
)
def test_get_document_status_document_pending(pact, status):
    response = {
        "doc_name": sample_doc_name,
        "status": status,
    }

    (
        pact.upon_receiving(f"Get status for {status} document")
        .given(f"Document {sample_hash} is {status}")
        .with_request("GET", f"/documents/{sample_hash}/status")
        .will_respond_with(200)
        .with_body(response)
    )

    with pact.serve() as srv:
        dms_client = DocumentManagementClient(srv.url)

        response = dms_client.get_document_status(sample_hash)
        assert response == status


@pytest.mark.parametrize(
    "status",
    [DocumentStatus.PENDING, DocumentStatus.COMPLETED, DocumentStatus.ERROR],
)
def test_update_document_status_already_existing_returns_success(pact, status):
    request = {"status": status}

    response = {
        "doc_hash": sample_hash,
        "doc_name": sample_doc_name,
        "status": status,
    }

    (
        pact.upon_receiving(f"Request to update existing document status to {status}")
        .given(f"Document {sample_hash} already exists in the db")
        .with_request("PUT", f"/documents/{sample_hash}/status")
        .with_body(request)
        .will_respond_with(204)
        .with_body(response)
    )

    with pact.serve() as srv:
        dms_client = DocumentManagementClient(srv.url)

        response = dms_client.update_document_status(sample_hash, status)
        assert response is None


@pytest.mark.parametrize(
    "status",
    [DocumentStatus.PENDING, DocumentStatus.COMPLETED, DocumentStatus.ERROR],
)
def test_update_document_status_not_existing_returns_success(pact, status):
    request = {"status": status}

    response = {
        "doc_hash": sample_hash,
        "doc_name": sample_doc_name,
        "status": status,
    }

    (
        pact.upon_receiving(f"Request to update new document status to {status}")
        .given(f"Document {sample_hash} does not exist in the db")
        .with_request("PUT", f"/documents/{sample_hash}/status")
        .with_body(request)
        .will_respond_with(201)
        .with_body(response)
    )

    with pact.serve() as srv:
        dms_client = DocumentManagementClient(srv.url)

        response = dms_client.update_document_status(sample_hash, status)
        assert response is None


def test_get_ingested_documents_returns_list(pact):
    response = [
        {
            "doc_hash": sample_hash,
            "doc_name": sample_doc_name,
            "status": DocumentStatus.PENDING,
        },
        {
            "doc_hash": "Doc Hash 2",
            "doc_name": "Doc Name 2",
            "status": DocumentStatus.COMPLETED,
        },
    ]
    (
        pact.upon_receiving(
            "Request to get processed documents and DMS has processed docs"
        )
        .given("DMS has documents registered")
        .with_request("GET", "/documents")
        .will_respond_with(200)
        .with_body(response)
    )
    with pact.serve() as srv:
        dms_client = DocumentManagementClient(srv.url)

        assert dms_client.get_documents() == [DMSDocument(**item) for item in response]


def test_get_ingested_documents_returns_empty(pact):
    (
        pact.upon_receiving(
            "Request to get processed documents and DMS has no processed docs"
        )
        .given("DMS has no documents registered")
        .with_request("GET", "/documents")
        .will_respond_with(204)
    )
    with pact.serve() as srv:
        dms_client = DocumentManagementClient(srv.url)

        assert dms_client.get_documents() == []
