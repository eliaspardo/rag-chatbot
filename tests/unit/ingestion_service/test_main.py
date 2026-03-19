from contextlib import asynccontextmanager, contextmanager

from fastapi.testclient import TestClient
from requests import HTTPError
from src.ingestion_service.main import SingleIngestionRequest, health
from unittest.mock import Mock, patch

from src.shared.constants import DocumentStatus
from src.shared.exceptions import NoDocumentsException
from src.shared.models import DMSDocument

from src.ingestion_service import main as api_main

document_request = SingleIngestionRequest(document="test")


@asynccontextmanager
async def _no_lifespan(app):
    yield


@contextmanager
def _build_client_no_lifespan():
    original_lifespan = api_main.app.router.lifespan_context
    api_main.app.router.lifespan_context = _no_lifespan
    try:
        with TestClient(api_main.app) as client:
            yield client
    finally:
        api_main.app.router.lifespan_context = original_lifespan


class TestMain:
    @patch("src.ingestion_service.main.get_vectordb_collection_count")
    @patch("src.ingestion_service.main.get_dms_documents")
    def test_health(
        self,
        mock_get_dms_documents,
        mock_get_vectordb_collection_count,
    ):
        mock_get_vectordb_collection_count.return_value = 2
        dms_response = [
            DMSDocument(
                doc_hash="Doc Hash 1",
                doc_name="Doc Name 1",
                status=DocumentStatus.PENDING,
            ),
            DMSDocument(
                doc_hash="Doc Hash 2",
                doc_name="Doc Name 2",
                status=DocumentStatus.ERROR,
            ),
        ]
        mock_get_dms_documents.return_value = dms_response
        result = health()
        assert result["status"] == "ok"
        assert result["documents_loaded_in_vector_store"] == "2"
        assert result["documents_loaded_in_dms"] == [
            doc.model_dump() for doc in dms_response
        ]

    @patch("src.ingestion_service.main.get_vectordb_collection_count")
    @patch("src.ingestion_service.main.get_dms_documents")
    def test_health_no_documents(
        self,
        mock_get_dms_documents,
        mock_get_vectordb_collection_count,
    ):
        mock_get_vectordb_collection_count.return_value = 0
        mock_get_dms_documents.return_value = []
        result = health()
        assert result["status"] == "ok"
        assert result["documents_loaded_in_vector_store"] == "0"
        assert result["documents_loaded_in_dms"] == []

    def test_ingest_document_404_no_document_found(self):
        mock_doc_ingestor = Mock()
        api_main.app.state.doc_ingestor = mock_doc_ingestor
        api_main.app.state.doc_ingestor.ingest_document.side_effect = (
            NoDocumentsException()
        )
        with _build_client_no_lifespan() as client:
            response = client.post(
                "/ingestion/document/", json=document_request.model_dump()
            )

        assert response.status_code == 404
        assert response.json()["detail"] == "File or documents not found"

    def test_ingest_document_503_cannot_connect_with_DMS(self):
        mock_doc_ingestor = Mock()
        api_main.app.state.doc_ingestor = mock_doc_ingestor
        api_main.app.state.doc_ingestor.ingest_document.side_effect = HTTPError()
        with _build_client_no_lifespan() as client:
            response = client.post(
                "/ingestion/document/", json=document_request.model_dump()
            )

        assert response.status_code == 503
        assert response.json()["detail"] == "Error calling DMS"

    def test_ingest_document_500_error(self):
        mock_doc_ingestor = Mock()
        api_main.app.state.doc_ingestor = mock_doc_ingestor
        api_main.app.state.doc_ingestor.ingest_document.side_effect = Exception()
        with _build_client_no_lifespan() as client:
            response = client.post(
                "/ingestion/document/", json=document_request.model_dump()
            )

        assert response.status_code == 500
        assert response.json()["detail"] == "Processing failed"

    def test_ingest_document_success(self):
        mock_doc_ingestor = Mock()
        api_main.app.state.doc_ingestor = mock_doc_ingestor
        api_main.app.state.doc_ingestor.ingest_document.side_effect = None
        with _build_client_no_lifespan() as client:
            response = client.post(
                "/ingestion/document/", json=document_request.model_dump()
            )

        assert response.status_code == 200
        assert (
            response.json()["message"]
            == "Document processed and saved to vector store!"
        )
