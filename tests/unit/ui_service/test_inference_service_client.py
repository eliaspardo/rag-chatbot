import os
from unittest.mock import Mock, patch

import pytest
import requests

from src.ui_service.inference_service_client import (
    ChatResponse,
    DocumentInfo,
    InferenceServiceClient,
)

CHAT_TIMEOUT = int(os.getenv("CHAT_TIMEOUT", "120"))


@pytest.fixture
def client():
    return InferenceServiceClient("http://localhost:8000")


class TestGetHealth:
    def test_get_health_success(self, client):
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
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            result = client.get_health()

        assert result.is_healthy is True
        assert result.vector_store_count == 2
        assert len(result.documents) == 2
        assert result.documents[0] == DocumentInfo(
            doc_hash="abc123",
            doc_name="doc1.pdf",
            status="Document processing completed",
        )
        assert result.documents[1] == DocumentInfo(
            doc_hash="def456",
            doc_name="doc2.pdf",
            status="Document pending processing",
        )

    def test_get_health_connection_error(self, client):
        with patch("requests.get", side_effect=requests.ConnectionError()):
            result = client.get_health()

        assert result.is_healthy is False
        assert result.error_message == "Inference Service: unreachable"

    def test_get_health_timeout(self, client):
        with patch("requests.get", side_effect=requests.Timeout()):
            result = client.get_health()

        assert result.is_healthy is False
        assert result.error_message == "Health check timeout"

    def test_get_health_invalid_json(self, client):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with patch("requests.get", return_value=mock_response):
            result = client.get_health()

        assert result.is_healthy is False

    def test_get_health_no_documents(self, client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "ok",
            "documents_loaded_in_vector_store": "0",
            "documents_loaded_in_dms": [],
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            result = client.get_health()

        assert result.is_healthy is True
        assert result.vector_store_count == 0
        assert result.documents == []


class TestAskQuestion:
    def test_ask_question_success(self, client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "answer": "Paris",
            "session_id": "session-123",
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = client.ask_question("What is the capital of France?")

        assert result == ChatResponse(answer="Paris", session_id="session-123")
        mock_post.assert_called_once_with(
            "http://localhost:8000/chat/domain-expert/",
            json={"question": "What is the capital of France?", "session_id": None},
            timeout=CHAT_TIMEOUT,
        )

    def test_ask_question_with_session_id(self, client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "answer": "Paris",
            "session_id": "session-123",
            "system_message": "Session resumed",
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = client.ask_question("What is the capital?", "session-123")

        assert result.session_id == "session-123"
        assert result.system_message == "Session resumed"
        mock_post.assert_called_once_with(
            "http://localhost:8000/chat/domain-expert/",
            json={"question": "What is the capital?", "session_id": "session-123"},
            timeout=CHAT_TIMEOUT,
        )

    def test_ask_question_error(self, client):
        with patch(
            "requests.post", side_effect=requests.RequestException("Connection refused")
        ):
            with pytest.raises(requests.RequestException):
                client.ask_question("What is the capital?")
