from src.ingestion_service.main import health
from unittest.mock import patch

from src.shared.constants import DocumentStatus
from src.shared.models import DMSDocument


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
