from unittest.mock import Mock, patch
from fastapi import HTTPException
from pydantic import ValidationError
import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.document_management_service.db_client import DBClient
from src.document_management_service.main import (
    get_document_status,
    get_documents,
    put_document_status,
)
from src.shared.constants import DocumentStatus
from src.shared.exceptions import DocumentHashConflictException

sample_hash = "d41d8cd98f00b204e9800998ecf8427e"
sample_doc_name = "Test doc name"
sample_status = DocumentStatus.PENDING


class TestMain:
    @pytest.fixture()
    def db_client(self):
        return Mock(spec=DBClient)

    @patch("src.document_management_service.main.app.state")
    def test_get_document_status_db_error_get_document_name(self, state, db_client):
        state.db_client = db_client
        db_client.get_document_name.side_effect = SQLAlchemyError()
        with pytest.raises(HTTPException) as exc_info:
            get_document_status(sample_hash)
        assert exc_info.value.status_code == 503

    @patch("src.document_management_service.main.app.state")
    def test_get_document_status_db_error_get_document_status(self, state, db_client):
        state.db_client = db_client
        db_client.get_document_name.return_value = sample_doc_name
        db_client.get_document_status.side_effect = SQLAlchemyError()
        with pytest.raises(HTTPException) as exc_info:
            get_document_status(sample_hash)
        assert exc_info.value.status_code == 503

    @patch("src.document_management_service.main.app.state")
    def test_put_document_status_conflict_error_set_document_status(
        self, state, db_client
    ):
        state.db_client = db_client
        db_client.set_document_status.side_effect = DocumentHashConflictException()
        mock_request = Mock()
        with pytest.raises(HTTPException) as exc_info:
            put_document_status(sample_hash, mock_request)
        assert exc_info.value.status_code == 503

    @patch("src.document_management_service.main.app.state")
    def test_put_document_status_db_error_set_document_status(self, state, db_client):
        state.db_client = db_client
        db_client.set_document_status.side_effect = SQLAlchemyError()
        mock_request = Mock()
        with pytest.raises(HTTPException) as exc_info:
            put_document_status(sample_hash, mock_request)
        assert exc_info.value.status_code == 503

    @patch("src.document_management_service.main.app.state")
    def test_put_document_status_validation_error_set_document_status(
        self, state, db_client
    ):
        state.db_client = db_client
        db_client.set_document_status.side_effect = ValidationError.from_exception_data(
            title="DMSDocument", line_errors=[]
        )
        mock_request = Mock()
        with pytest.raises(HTTPException) as exc_info:
            put_document_status(sample_hash, mock_request)
        assert exc_info.value.status_code == 503

    @patch("src.document_management_service.main.app.state")
    def test_get_documents_db_error(self, state, db_client):
        state.db_client = db_client
        db_client.get_documents.side_effect = SQLAlchemyError()
        with pytest.raises(HTTPException) as exc_info:
            get_documents()
        assert exc_info.value.status_code == 503

    @patch("src.document_management_service.main.app.state")
    def test_get_documents_validation_error(self, state, db_client):
        state.db_client = db_client
        db_client.get_documents.side_effect = ValidationError.from_exception_data(
            title="DMSDocument", line_errors=[]
        )
        with pytest.raises(HTTPException) as exc_info:
            get_documents()
        assert exc_info.value.status_code == 503
