import hashlib
import os
import chromadb
import responses
import pytest
from unittest.mock import patch
from testcontainers.core.container import DockerContainer
from fastapi.testclient import TestClient

from src.ingestion_service.main import IngestionRequest, SingleIngestionRequest
from src.shared.constants import DocumentStatus
from tests.integration.helpers import (
    extract_doc_name,
    seed_chromadb_documents,
)

os.environ["DOCKER_HOST"] = "unix:///home/eliaspardo/.docker/desktop/docker.sock"
os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"

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
doc_hash_non_existing = hashlib.md5(document_path_non_existing.encode()).hexdigest()
doc_name__non_existing = extract_doc_name(document_path_non_existing)
document_request__non_existing = SingleIngestionRequest(
    document=document_path_non_existing
)


@pytest.fixture(scope="class")
def chroma_container():
    """
    Start a temporary ChromaDB Docker container for use by tests.
    
    Yields:
        DockerContainer: A running ChromaDB container exposing the service port; the container is stopped when the fixture is torn down.
    """
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
    """
    Prepare and apply environment variables required for integration tests and provide the resulting mapping.
    
    The fixture sets Chroma connection details, a temporary AWS folder, DMS URL (mock target), model and preprocessing settings, and ingestion chunking parameters, and patches os.environ with these values for the duration of the fixture.
    
    Returns:
        env_vars (dict): Mapping of environment variable names to values as applied to os.environ.
    """
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
def mock_dms(integration_env):
    """
    Provide a configured RequestsMock that intercepts DMS HTTP calls while allowing external hosts required by integration tests to pass through.
    
    Parameters:
        integration_env (dict): Environment values containing 'CHROMA_HOST' and 'CHROMA_PORT' used to compute the ChromaDB base URL that should be excluded from interception.
    
    Returns:
        responses.RequestsMock: A RequestsMock instance configured to permit passthrough for HuggingFace, the ChromaDB base URL derived from `integration_env`, and Docker API requests.
    """
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
    """
    Provide a TestClient for the FastAPI ingestion app configured with the integration test environment.
    
    This fixture clears any previously imported `src.*` modules so the application is re-imported and reads the patched environment variables supplied by the integration fixtures.
    
    Parameters:
    	integration_env (dict): Environment variables prepared for integration tests.
    	mock_dms (responses.RequestsMock): Active HTTP mock for DMS interactions.
    
    Returns:
    	test_client (fastapi.testclient.TestClient): A TestClient instance for `src.ingestion_service.main.app`.
    """
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
        def status_callback(request):
            """
            Handle mocked DMS PUT callbacks by inspecting the request JSON `status` and returning the corresponding HTTP response for tests.
            
            Parameters:
                request: The incoming HTTP request object from the `responses` mock; its body is expected to be JSON containing a `status` field matching `DocumentStatus`.
            
            Returns:
                A tuple (status_code, headers, body) where:
                - If `status` is `DocumentStatus.PENDING`: returns status 201, empty headers, and a JSON body containing the `dms_documents_pending` payload.
                - If `status` is `DocumentStatus.COMPLETED`: returns status 204, empty headers, and an empty body.
            """
            import json

            body = json.loads(request.body)

            if body["status"] == DocumentStatus.PENDING:
                return (201, {}, json.dumps(dms_documents_pending))
            elif body["status"] == DocumentStatus.COMPLETED:
                return (204, {}, "")

        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash}/status/",
            callback=status_callback,
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
        """
        Verifies that ingesting two documents results in both documents being stored in the vector store and reported as completed by the DMS.
        
        Mocks per-document DMS status checks and status-update endpoints to simulate PENDING→COMPLETED transitions for two distinct documents, invokes the bulk ingestion endpoint, then asserts the service /health endpoint reports two documents in the vector store and the DMS listing contains both documents with `COMPLETED` status.
        """
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
        def status_callback(request):
            """
            Handle mocked DMS PUT callbacks by inspecting the request JSON `status` and returning the corresponding HTTP response for tests.
            
            Parameters:
                request: The incoming HTTP request object from the `responses` mock; its body is expected to be JSON containing a `status` field matching `DocumentStatus`.
            
            Returns:
                A tuple (status_code, headers, body) where:
                - If `status` is `DocumentStatus.PENDING`: returns status 201, empty headers, and a JSON body containing the `dms_documents_pending` payload.
                - If `status` is `DocumentStatus.COMPLETED`: returns status 204, empty headers, and an empty body.
            """
            import json

            body = json.loads(request.body)

            if body["status"] == DocumentStatus.PENDING:
                return (201, {}, json.dumps(dms_documents_pending))
            elif body["status"] == DocumentStatus.COMPLETED:
                return (204, {}, "")

        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash}/status/",
            callback=status_callback,
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

        def status_callback_2(request):
            """
            Handle a mocked DMS PUT request for a document's status and return the corresponding HTTP response tuple.
            
            Parameters:
                request: The incoming HTTP request object whose .body is a JSON-encoded dict containing a "status" key.
            
            Returns:
                A tuple (status_code, headers, body):
                  - If `status` is `DocumentStatus.PENDING`: returns status 201, empty headers, and a JSON payload representing the pending document list.
                  - If `status` is `DocumentStatus.COMPLETED`: returns status 204, empty headers, and an empty body.
            """
            import json

            body = json.loads(request.body)

            if body["status"] == DocumentStatus.PENDING:
                return (201, {}, json.dumps(dms_documents_pending_2))
            elif body["status"] == DocumentStatus.COMPLETED:
                return (204, {}, "")

        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash_2}/status/",
            callback=status_callback_2,
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
        """
        Integration test that ingests two documents where one fails and verifies health reporting.
        
        Sets up DMS mocks so the first document transitions to `ERROR` and the second to `COMPLETED`, performs a POST to `/ingestion/documents`, then checks `/health` to confirm the vector store contains one document and the DMS listing contains the completed and error entries.
        """
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
                "doc_name": doc_name__non_existing,
                "status": DocumentStatus.PENDING,
            },
        ]
        dms_documents_error_non_existing = [
            {
                "doc_hash": doc_hash_non_existing,
                "doc_name": doc_name__non_existing,
                "status": DocumentStatus.ERROR,
            },
        ]

        # Responses mocking - send response based on request's status
        def status_callback(request):
            """
            Handle a mocked DMS status update callback that responds based on the request's `status` field.
            
            Parameters:
                request: HTTP request object whose body is JSON and contains a `status` field set to a `DocumentStatus` value.
            
            Returns:
                tuple: (status_code, headers, body) where
                    - for `DocumentStatus.PENDING`: returns status 201, empty headers, and a JSON string payload containing the pending document list (`dms_documents_pending_non_existing`);
                    - for `DocumentStatus.ERROR`: returns status 204, empty headers, and an empty body.
            """
            import json

            body = json.loads(request.body)

            if body["status"] == DocumentStatus.PENDING:
                return (201, {}, json.dumps(dms_documents_pending_non_existing))
            elif body["status"] == DocumentStatus.ERROR:
                return (204, {}, "")

        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash_non_existing}/status/",
            callback=status_callback,
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

        def status_callback_2(request):
            """
            Handle a mocked DMS PUT request for a document's status and return the corresponding HTTP response tuple.
            
            Parameters:
                request: The incoming HTTP request object whose .body is a JSON-encoded dict containing a "status" key.
            
            Returns:
                A tuple (status_code, headers, body):
                  - If `status` is `DocumentStatus.PENDING`: returns status 201, empty headers, and a JSON payload representing the pending document list.
                  - If `status` is `DocumentStatus.COMPLETED`: returns status 204, empty headers, and an empty body.
            """
            import json

            body = json.loads(request.body)

            if body["status"] == DocumentStatus.PENDING:
                return (201, {}, json.dumps(dms_documents_pending_2))
            elif body["status"] == DocumentStatus.COMPLETED:
                return (204, {}, "")

        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash_2}/status/",
            callback=status_callback_2,
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
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["documents_loaded_in_vector_store"] == "0"
        assert data["documents_loaded_in_dms"] == []

    def test_ingestion_1_document_vector_store_unavailable(
        self, client, mock_dms, integration_env, chroma_container
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
        def status_callback(request):
            """
            Handle a mocked DMS status update request and return the appropriate HTTP response tuple.
            
            Parameters:
                request: The incoming HTTP request object containing a JSON body with a "status" field.
            
            Returns:
                A tuple (status_code, headers, body):
                - (201, {}, <json>) when the request body "status" equals DocumentStatus.PENDING; the body is the JSON-serialized pending documents list.
                - (204, {}, "") when the request body "status" equals DocumentStatus.ERROR.
            """
            import json

            body = json.loads(request.body)

            if body["status"] == DocumentStatus.PENDING:
                return (201, {}, json.dumps(dms_documents_pending))
            elif body["status"] == DocumentStatus.ERROR:
                return (204, {}, "")

        mock_dms.add_callback(
            responses.PUT,
            f"http://localhost:8004/documents/{doc_hash}/status/",
            callback=status_callback,
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
