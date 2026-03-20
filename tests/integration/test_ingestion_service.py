import os
import chromadb
import responses
import pytest
from unittest.mock import patch
from testcontainers.core.container import DockerContainer
from fastapi.testclient import TestClient

from src.shared.constants import DocumentStatus
from tests.integration.helpers import seed_chromadb_documents

os.environ["DOCKER_HOST"] = "unix:///home/eliaspardo/.docker/desktop/docker.sock"
os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"


@pytest.fixture(scope="class")
def chroma_container():
    """Spin up ChromaDB testcontainer for the test class."""
    container = (
        DockerContainer("chromadb/chroma:1.5.1.dev68")
        .with_exposed_ports(8000)
        .with_env("IS_PERSISTENT", "FALSE")
        .with_env("ANONYMIZED_TELEMETRY", "FALSE")
    )

    container.start()

    yield container

    container.stop()


@pytest.fixture(scope="class")
def integration_env(chroma_container, tmp_path_factory):
    """Configure environment variables for integration tests."""
    host = chroma_container.get_container_host_ip()
    port = chroma_container.get_exposed_port(8000)
    temp_dir = tmp_path_factory.mktemp("aws_temp")

    env_vars = {
        "CHROMA_HOST": host,
        "CHROMA_PORT": str(port),
        "CHROMA_COLLECTION": "test_collection",
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
def mock_dms():
    """Mock DMS HTTP responses using responses library."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        # Allow HuggingFace requests to pass through (for model downloads)
        rsps.add_passthru("https://huggingface.co")
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


@pytest.fixture
def chromadb_client(integration_env):
    """Fixture that provides a ChromaDB client"""
    chroma_client = chromadb.HttpClient(
        host=integration_env["CHROMA_HOST"], port=int(integration_env["CHROMA_PORT"])
    )

    yield chroma_client

    # Cleanup: delete collection after test
    try:
        chroma_client.delete_collection(integration_env["CHROMA_COLLECTION"])
    except Exception:
        pass


class TestIngestionService:
    def test_health_check_with_no_documents(self, client, mock_dms):
        # Mock DMS /documents/ endpoint (called by health check)
        mock_dms.add(
            responses.GET,
            "http://localhost:8004/documents/",
            json=[],  # No documents
            status=200,
        )

        response = client.get("/health")

        # Assert: Service is healthy
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["documents_loaded_in_vector_store"] == "0"
        assert data["documents_loaded_in_dms"] == []

    def test_health_check_with_documents(
        self, client, mock_dms, chromadb_client, integration_env
    ):
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
        # Mock DMS /documents/ endpoint (called by health check)
        mock_dms.add(
            responses.GET,
            "http://localhost:8004/documents/",
            json=dms_documents,  # No documents
            status=200,
        )

        response = client.get("/health")

        # Assert: Service is healthy and returning expected documents
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["documents_loaded_in_vector_store"] == "1"
        assert data["documents_loaded_in_dms"] == dms_documents
