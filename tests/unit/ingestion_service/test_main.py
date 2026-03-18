from src.ingestion_service.main import health
from unittest.mock import patch


class TestMain:
    @patch("src.ingestion_service.main.get_vectordb_collection_count")
    def test_health(self, mock_get_vectordb_collection_count):
        mock_get_vectordb_collection_count.return_value = 3
        result = health()
        assert result["status"] == "ok"
        assert result["documents_loaded"] == "3"
