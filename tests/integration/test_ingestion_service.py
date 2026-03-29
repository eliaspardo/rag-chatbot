import hashlib
import os
import chromadb
import responses
import pytest
from unittest.mock import patch
from testcontainers.core.container import DockerContainer
from testcontainers.localstack import LocalStackContainer
from fastapi.testclient import TestClient

from src.ingestion_service.main import IngestionRequest, SingleIngestionRequest
from src.shared.constants import DocumentStatus
from tests.integration.helpers import (
    extract_doc_name,
    seed_chromadb_documents,
)

document_path = "tests/data/pdf-test.pdf"
doc_hash = hashlib.md5(document_path.encode()).hexdigest()
doc_name = extract_doc_name(document_path)
document_request = SingleIngestionRequest(document=document_path)

document_path_2 = "tests/data/pdf-test-2.pdf"
doc_hash_2 = hashlib.md5(document_path_2.encode()).hexdigest()
doc_name_2 = extract_doc_name(document_path_2)
document_request_2 = SingleIngestionRequest(document=document_path_2)

documents_request = IngestionRequest(documents=[document_path, document_path_2])

document_path_non_existing = "tests/data/pdf-test-non-existing.pdf"


def make_status_callback(pending_response, terminal_status):
    """Factory for DMS status update callbacks.

    Creates a callback that:
    - Returns 201 with pending_response when status is PENDING
    - Returns 204 (no content) when status matches terminal_status (COMPLETED or ERROR)
    """

    def callback(request):
        import json

        body = json.loads(request.body)

        if body["status"] == DocumentStatus.PENDING:
            return (201, {}, json.dumps(pending_response))
        elif body["status"] == terminal_status:
            return (204, {}, "")

    return callback


doc_hash_non_existing = hashlib.md5(document_path_non_existing.encode()).hexdigest()
doc_name_non_existing = extract_doc_name(document_path_non_existing)
document_request_non_existing = SingleIngestionRequest(
    document=document_path_non_existing
)


@pytest.fixture(scope="class")
def chroma_container():
    """Spin up ChromaDB testcontainer for the test class."""
    container = (
        DockerContainer("chromadb/chroma:1.5.1.dev68")
        .with_bind_ports(8000, 8765)  # Bind container port 8000 to host port 8765
        .with_env("IS_PERSISTENT", "FALSE")
        .with_env("ANONYMIZED_TELEMETRY", "FALSE")
    )

    container.start()

    yield container

    container.stop()


@pytest.fixture(scope="class")
def localstack_container():
    with LocalStackContainer(image="gresau/localstack-persist:latest").with_env(
        "IS_PERSISTENT", "FALSE"
    ).with_env("ANONYMIZED_TELEMETRY", "FALSE") as localstack:
        yield localstack


@pytest.fixture(scope="class")
def s3_client(localstack_container):
    """Create S3 client and seed with test data."""
    import boto3

    localstack_host = localstack_container.get_container_host_ip()
    localstack_port = localstack_container.get_exposed_port(4566)

    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{localstack_host}:{localstack_port}",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )

    # Create bucket and upload file
    s3.create_bucket(Bucket="sample-bucket")
    s3.upload_file(
        "tests/data/pdf-test.pdf",
        "sample-bucket",
        "pdf-test.pdf",
    )

    yield s3


@pytest.fixture(scope="class")
def integration_env(chroma_container, tmp_path_factory, localstack_container):
    """Configure environment variables for integration tests."""
    host = chroma_container.get_container_host_ip()
    port = chroma_container.get_exposed_port(8000)
    localstack_host = localstack_container.get_container_host_ip()
    localstack_port = localstack_container.get_exposed_port(4566)
    temp_dir = tmp_path_factory.mktemp("aws_temp")

    env_vars = {
        "CHROMA_HOST": host,
        "CHROMA_PORT": str(port),
        "CHROMA_COLLECTION": "test_collection",
        "AWS_ENDPOINT_URL": f"http://{localstack_host}:{localstack_port}",
        "DMS_URL": "http://localhost:8004",  # Will be mocked
        "AWS_TEMP_FOLDER": str(temp_dir),
        "PDF_PATH": "",  # Empty for happy path
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "RAG_PREPROCESSOR": "legacy",
        "CHUNK_SIZE": "1500",
        "CHUNK_OVERLAP": "150",
    }

    # Apply environment variables
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def mock_dms(integration_env):
    """Mock DMS HTTP responses using responses library."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        # Allow HuggingFace requests to pass through (for model downloads)
        rsps.add_passthru("https://huggingface.co")
        # Allow ChromaDB requests to pass through (so we can test real container failures)
        chroma_url = (
            f"http://{integration_env['CHROMA_HOST']}:{integration_env['CHROMA_PORT']}"
        )
        rsps.add_passthru(chroma_url)
        # Allow Docker API requests to pass through (for testcontainers management)
        rsps.add_passthru("http+docker://")
        yield rsps


@pytest.fixture
def client(integration_env, mock_dms):
    """Create FastAPI TestClient with integration dependencies."""
    import sys

    # Clear any cached imports so modules read the new env vars
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith("src.")]
    for module in modules_to_clear:
        del sys.modules[module]

    # NOW import - modules will read integration_env values
    from src.ingestion_service.main import app

    with TestClient(app) as test_client:
        yield test_client


# Setting autoreuse to true to cleanup ChromaDB after each test
@pytest.fixture(autouse=True)
def chromadb_client(integration_env):
    """Fixture that provides a ChromaDB client"""
    chroma_client = chromadb.HttpClient(
        host=integration_env["CHROMA_HOST"], port=int(integration_env["CHROMA_PORT"])
    )

    yield chroma_client

    # Cleanup: delete collection after test
    try:
        chroma_client = chromadb.HttpClient(
            host=integration_env["CHROMA_HOST"],
            port=int(integration_env["CHROMA_PORT"]),
        )
        chroma_client.delete_collection(integration_env["CHROMA_COLLECTION"])
    except Exception:
        pass


class TestIngestionService:
    def test_health_check_with_no_documents(self, client, mock_dms):
        #
        # Arrange - Mock DMS documents endpoint - no documents
        #
        mock_dms.add(
            responses.GET,
            "http://localhost:8004/documents/",
            json=[],  # No documents
            status=200,
        )

        #
        # Act - Get health
        #
        response = client.get("/health")

        #
        # Assert
        #
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["documents_loaded_in_vector_store"] == "0"
        assert data["documents_loaded_in_dms"] == []

    def test_health_check_with_documents(
        self, client, mock_dms, chromadb_client, integration_env
    ):
        #
        # Arrange - Seed ChromaDB Get Health, Mock DMS documents endpoint
        #
        seed_chromadb_documents(
            chroma_client=chromadb_client,
            collection_name=integration_env["CHROMA_COLLECTION"],
            texts=["Sample RAG document about testing"],
            metadatas=[{"source": "test.pdf"}],
        )

        dms_documents = [
            {
                "doc_hash": "Doc Hash 1",
                "doc_name": "Doc Name 1",
                "status": DocumentStatus.PENDING,
            },
        ]
        mock_dms.add(
            responses.GET,
            "http://localhost:8004/documents/",
            json=dms_documents,
            status=200,
        )

        #
        # Act - Get Health
        #
        response = client.get("/health")

        #
        # Assert: Service is healthy and returning expected documents
        #
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["documents_loaded_in_vector_store"] == "1"
        assert data["documents_loaded_in_dms"] == dms_documents

    def test_ingestion_1_document(self, client, mock_dms, integration_env):
        #
        # Arrange - Mock DMS endpoint: not found, pending, completed, return documents
        #
        mock_dms.add(
            responses.GET,
            f"http://localhost:8004/documents/{doc_hash}/status/",
            status=404,
        )
        dms_documents_pending = [
            {
                "doc_hash": doc_hash,
                "doc_name": doc_name,
                "status": DocumentStatus.PENDING,
            },
        ]
        dms_documents_completed = [
            {
                "doc_hash": doc_hash,
                "doc_name": doc_name,
                "status": DocumentStatus.COMPLETED,
            },
        ]

        # Responses mocking - send response based on request's status
        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash}/status/",
            callback=make_status_callback(
                dms_documents_pending, DocumentStatus.COMPLETED
            ),
        )
        # After document has been added, return document - used by health check for assertion
        mock_dms.add(
            responses.GET,
            "http://localhost:8004/documents/",
            json=dms_documents_completed,
            status=200,
        )

        #
        # Act - Request single document ingestion
        #
        response = client.post(
            "/ingestion/document", json=document_request.model_dump()
        )

        #
        # Assert
        #
        assert response.status_code == 200
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["documents_loaded_in_vector_store"] == "1"
        assert data["documents_loaded_in_dms"] == dms_documents_completed

    def test_ingestion_2_documents(self, client, mock_dms, integration_env):
        #
        # Arrange - Mock DMS endpoint: not found, pending, completed, return documents
        #
        mock_dms.add(
            responses.GET,
            f"http://localhost:8004/documents/{doc_hash}/status/",
            status=404,
        )
        mock_dms.add(
            responses.GET,
            f"http://localhost:8004/documents/{doc_hash_2}/status/",
            status=404,
        )
        dms_documents_pending = [
            {
                "doc_hash": doc_hash,
                "doc_name": doc_name,
                "status": DocumentStatus.PENDING,
            },
        ]
        dms_documents_completed = [
            {
                "doc_hash": doc_hash,
                "doc_name": doc_name,
                "status": DocumentStatus.COMPLETED,
            },
        ]

        # Responses mocking - send response based on request's status
        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash}/status/",
            callback=make_status_callback(
                dms_documents_pending, DocumentStatus.COMPLETED
            ),
        )
        dms_documents_pending_2 = [
            {
                "doc_hash": doc_hash_2,
                "doc_name": doc_name_2,
                "status": DocumentStatus.PENDING,
            },
        ]
        dms_documents_completed_2 = [
            {
                "doc_hash": doc_hash_2,
                "doc_name": doc_name_2,
                "status": DocumentStatus.COMPLETED,
            },
        ]

        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash_2}/status/",
            callback=make_status_callback(
                dms_documents_pending_2, DocumentStatus.COMPLETED
            ),
        )

        # After document has been added, return documents - used by health check for assertion
        mock_dms.add(
            responses.GET,
            "http://localhost:8004/documents/",
            json=dms_documents_completed + dms_documents_completed_2,
            status=200,
        )

        #
        # Act - Request multiple document ingestion
        #
        response = client.post(
            "/ingestion/documents",
            json=documents_request.model_dump(),
        )

        #
        # Assert
        #
        assert response.status_code == 200
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["documents_loaded_in_vector_store"] == "2"
        assert (
            data["documents_loaded_in_dms"]
            == dms_documents_completed + dms_documents_completed_2
        )

    def test_ingestion_2_documents_1_error(self, client, mock_dms, integration_env):
        #
        # Arrange - Mock DMS endpoint: not found, pending, completed, return document (one non existing doc)
        #
        mock_dms.add(
            responses.GET,
            f"http://localhost:8004/documents/{doc_hash_non_existing}/status/",
            status=404,
        )
        mock_dms.add(
            responses.GET,
            f"http://localhost:8004/documents/{doc_hash_2}/status/",
            status=404,
        )
        dms_documents_pending_non_existing = [
            {
                "doc_hash": doc_hash_non_existing,
                "doc_name": doc_name_non_existing,
                "status": DocumentStatus.PENDING,
            },
        ]
        dms_documents_error_non_existing = [
            {
                "doc_hash": doc_hash_non_existing,
                "doc_name": doc_name_non_existing,
                "status": DocumentStatus.ERROR,
            },
        ]

        # Responses mocking - send response based on request's status
        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash_non_existing}/status/",
            callback=make_status_callback(
                dms_documents_pending_non_existing, DocumentStatus.ERROR
            ),
        )
        dms_documents_pending_2 = [
            {
                "doc_hash": doc_hash_2,
                "doc_name": doc_name_2,
                "status": DocumentStatus.PENDING,
            },
        ]
        dms_documents_completed_2 = [
            {
                "doc_hash": doc_hash_2,
                "doc_name": doc_name_2,
                "status": DocumentStatus.COMPLETED,
            },
        ]

        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash_2}/status/",
            callback=make_status_callback(
                dms_documents_pending_2, DocumentStatus.COMPLETED
            ),
        )

        # After document has been added, return documents - used by health check for assertion
        mock_dms.add(
            responses.GET,
            "http://localhost:8004/documents/",
            json=dms_documents_completed_2 + dms_documents_error_non_existing,
            status=200,
        )

        #
        # Act - Request multiple document ingestion
        #
        documents_request = IngestionRequest(
            documents=[document_path_non_existing, document_path_2]
        )
        response = client.post(
            "/ingestion/documents",
            json=documents_request.model_dump(),
        )

        #
        # Assert
        #
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["documents_loaded_in_vector_store"] == "1"
        assert (
            data["documents_loaded_in_dms"]
            == dms_documents_completed_2 + dms_documents_error_non_existing
        )

    def test_ingestion_1_document_DMS_unavailable(
        self, client, mock_dms, integration_env
    ):
        #
        # Arrange - Mock DMS endpoint: not found, pending, completed, return documents
        #
        mock_dms.add(
            responses.GET,
            f"http://localhost:8004/documents/{doc_hash}/status/",
            status=503,
        )

        mock_dms.add(
            responses.GET,
            "http://localhost:8004/documents/",
            status=503,
        )

        #
        # Act - Request single document ingestion
        #
        response = client.post(
            "/ingestion/document", json=document_request.model_dump()
        )

        #
        # Assert
        #
        assert response.status_code == 503
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["documents_loaded_in_vector_store"] == "0"
        assert data["documents_loaded_in_dms"] == []

    def test_ingestion_s3_document(self, client, mock_dms, integration_env, s3_client):
        s3_document_path = "s3://sample-bucket/pdf-test.pdf"
        s3_doc_hash = hashlib.md5(s3_document_path.encode()).hexdigest()
        s3_doc_name = extract_doc_name(s3_document_path)
        s3_document_request = SingleIngestionRequest(document=s3_document_path)
        #
        # Arrange - Mock DMS endpoint: not found, pending, completed, return documents
        #
        mock_dms.add(
            responses.GET,
            f"http://localhost:8004/documents/{s3_doc_hash}/status/",
            status=404,
        )
        s3_dms_documents_pending = [
            {
                "doc_hash": s3_doc_hash,
                "doc_name": s3_doc_name,
                "status": DocumentStatus.PENDING,
            },
        ]
        s3_dms_documents_completed = [
            {
                "doc_hash": s3_doc_hash,
                "doc_name": s3_doc_name,
                "status": DocumentStatus.COMPLETED,
            },
        ]

        # Responses mocking - send response based on request's status
        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{s3_doc_hash}/status/",
            callback=make_status_callback(
                s3_dms_documents_pending, DocumentStatus.COMPLETED
            ),
        )
        # After document has been added, return document - used by health check for assertion
        mock_dms.add(
            responses.GET,
            "http://localhost:8004/documents/",
            json=s3_dms_documents_completed,
            status=200,
        )

        #
        # Act - Request single document ingestion
        #
        response = client.post(
            "/ingestion/document", json=s3_document_request.model_dump()
        )

        #
        # Assert
        #
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["documents_loaded_in_vector_store"] == "1"
        assert data["documents_loaded_in_dms"] == s3_dms_documents_completed

    def test_ingestion_1_document_vector_store_unavailable(
        self,
        client,
        mock_dms,
        integration_env,
        chroma_container,
    ):
        #
        # Arrange - Mock DMS endpoint: not found, pending, completed, return documents
        #
        mock_dms.add(
            responses.GET,
            f"http://localhost:8004/documents/{doc_hash}/status/",
            status=404,
        )

        dms_documents_pending = [
            {
                "doc_hash": doc_hash,
                "doc_name": doc_name,
                "status": DocumentStatus.PENDING,
            },
        ]
        dms_documents_error = [
            {
                "doc_hash": doc_hash,
                "doc_name": doc_name,
                "status": DocumentStatus.ERROR,
            },
        ]

        # Responses mocking - send response based on request's status
        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash}/status/",
            callback=make_status_callback(dms_documents_pending, DocumentStatus.ERROR),
        )

        mock_dms.add(
            responses.GET,
            "http://localhost:8004/documents/",
            json=dms_documents_error,
            status=200,
        )

        # Stop chroma container to simulate unavailability
        chroma_container.stop()

        try:
            #
            # Act - Request single document ingestion
            #
            response = client.post(
                "/ingestion/document", json=document_request.model_dump()
            )

            #
            # Assert
            #
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["documents_loaded_in_vector_store"] == "0"
            assert data["documents_loaded_in_dms"] == dms_documents_error
        finally:
            # Restart container for subsequent tests
            chroma_container.start()
