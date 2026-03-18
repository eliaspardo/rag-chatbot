import pytest
import requests
from unittest.mock import Mock, patch

from src.ui_service.inference_service_client import (
    ChatResponse,
    DocumentInfo,
    HealthStatus,
    InferenceServiceClient,
)


class TestInferenceServiceClient:
    @pytest.fixture(scope="session")
    def client(self):
        return InferenceServiceClient("http://test-url")

    @patch("src.ui_service.inference_service_client.requests")
    def test_get_health_success(self, mock_requests, client):
        mock_response = Mock()
        mock_response.json.return_value = {
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
        mock_requests.get.return_value = mock_response

        result = client.get_health()

        assert result.is_healthy is True
        assert result.vector_store_count == 2
        assert len(result.documents) == 2
        assert result.documents[0] == DocumentInfo(
            doc_hash="abc123",
            doc_name="doc1.pdf",
            status="Document processing completed",
        )
        assert result.error_message is None

    @patch("src.ui_service.inference_service_client.requests")
    def test_get_health_connection_error(self, mock_requests, client):
        mock_requests.get.side_effect = requests.ConnectionError("Connection refused")
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.Timeout = requests.Timeout

        result = client.get_health()

        assert result.is_healthy is False
        assert result.error_message == "Inference Service: unreachable"

    @patch("src.ui_service.inference_service_client.requests")
    def test_get_health_timeout(self, mock_requests, client):
        mock_requests.get.side_effect = requests.Timeout("Timeout")
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.Timeout = requests.Timeout

        result = client.get_health()

        assert result.is_healthy is False
        assert result.error_message == "Health check timeout"

    @patch("src.ui_service.inference_service_client.requests")
    def test_get_health_invalid_json(self, mock_requests, client):
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_requests.get.return_value = mock_response
        mock_requests.ConnectionError = requests.ConnectionError
        mock_requests.Timeout = requests.Timeout

        result = client.get_health()

        assert result.is_healthy is False
        assert result.error_message is not None

    @patch("src.ui_service.inference_service_client.requests")
    def test_ask_question_success(self, mock_requests, client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "answer": "42",
            "session_id": "session-abc",
            "system_message": None,
        }
        mock_requests.post.return_value = mock_response

        result = client.ask_question("What is the answer?")

        assert result == ChatResponse(
            answer="42", session_id="session-abc", system_message=None
        )
        mock_requests.post.assert_called_once_with(
            "http://test-url/chat/domain-expert/",
            json={"question": "What is the answer?", "session_id": None},
            timeout=30,
        )

    @patch("src.ui_service.inference_service_client.requests")
    def test_ask_question_with_session_id(self, mock_requests, client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "answer": "Still 42",
            "session_id": "session-abc",
        }
        mock_requests.post.return_value = mock_response

        result = client.ask_question("Follow-up question?", session_id="session-abc")

        assert result.session_id == "session-abc"
        mock_requests.post.assert_called_once_with(
            "http://test-url/chat/domain-expert/",
            json={"question": "Follow-up question?", "session_id": "session-abc"},
            timeout=30,
        )

    @patch("src.ui_service.inference_service_client.requests")
    def test_ask_question_error(self, mock_requests, client):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Error")
        mock_requests.post.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            client.ask_question("What is the answer?")
