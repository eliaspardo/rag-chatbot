from typing import Generator
import pytest
from pact import Pact

from src.ui_service.inference_service_client import InferenceServiceClient


@pytest.fixture
def pact() -> Generator[Pact, None, None]:
    """Set up a Pact mock provider for consumer tests."""
    pact = Pact("ui-service", "inference-service").with_specification("V4")
    yield pact
    pact.write_file("pacts")


class TestInferenceHealth:
    def test_health_with_documents(self, pact):
        response_body = {
            "status": "ok",
            "documents_loaded_in_vector_store": "2",
            "documents_loaded_in_dms": [
                {
                    "doc_hash": "abc123",
                    "doc_name": "doc1.pdf",
                    "status": "Document processing completed",
                },
                {
                    "doc_hash": "def456",
                    "doc_name": "doc2.pdf",
                    "status": "Document pending processing",
                },
            ],
        }
        (
            pact.upon_receiving(
                "Get health when inference service has documents loaded"
            )
            .given("Inference service has documents loaded")
            .with_request("GET", "/health")
            .will_respond_with(200)
            .with_body(response_body)
        )

        with pact.serve() as srv:
            client = InferenceServiceClient(srv.url)
            result = client.get_health()

        assert result.is_healthy is True
        assert result.vector_store_count == 2
        assert len(result.documents) == 2
        assert result.documents[0].doc_name == "doc1.pdf"
        assert result.documents[1].doc_name == "doc2.pdf"

    def test_health_no_documents(self, pact):
        response_body = {
            "status": "ok",
            "documents_loaded_in_vector_store": "0",
            "documents_loaded_in_dms": [],
        }
        (
            pact.upon_receiving("Get health when inference service has no documents")
            .given("Inference service has no documents")
            .with_request("GET", "/health")
            .will_respond_with(200)
            .with_body(response_body)
        )

        with pact.serve() as srv:
            client = InferenceServiceClient(srv.url)
            result = client.get_health()

        assert result.is_healthy is True
        assert result.vector_store_count == 0
        assert result.documents == []
