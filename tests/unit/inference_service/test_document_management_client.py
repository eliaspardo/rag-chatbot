import pytest
from fastapi import Response
from unittest.mock import Mock, patch

import requests
from starlette.status import HTTP_204_NO_CONTENT


from src.inference_service.document_management_client import DocumentManagementClient
from src.shared.constants import DocumentStatus
from src.shared.models import DMSDocument


class TestDocumentManagementClient:
    @pytest.fixture(scope="session")
    def dms_client(self):
        return DocumentManagementClient("test_url")

    @patch("src.inference_service.document_management_client.requests")
    def test_get_documents_no_documents(self, mock_requests, dms_client):
        mock_requests.get.return_value = Response(status_code=HTTP_204_NO_CONTENT)

        result = dms_client.get_documents()
        assert result == []

    @patch("src.inference_service.document_management_client.requests")
    def test_get_documents_success(self, mock_requests, dms_client):
        response_data = [
            {
                "doc_hash": "Doc Hash 1",
                "doc_name": "Doc Name 1",
                "status": DocumentStatus.PENDING,
            },
        ]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_requests.get.return_value = mock_response

        result = dms_client.get_documents()
        assert result == [DMSDocument(**item) for item in response_data]

    @patch("src.inference_service.document_management_client.requests")
    def test_get_documents_error(self, mock_requests, dms_client):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error"
        )
        mock_requests.get.return_value = mock_response
        with pytest.raises(requests.HTTPError):
            dms_client.get_documents()

    @patch("src.inference_service.document_management_client.requests")
    def test_get_documents_conn_refused(self, mock_requests, dms_client):
        mock_requests.get.side_effect = requests.ConnectionError("Connection refused")
        with pytest.raises(Exception):
            dms_client.get_documents()
