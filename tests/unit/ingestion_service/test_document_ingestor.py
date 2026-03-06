from unittest.mock import ANY, Mock, call, patch
from pytest import fixture
import pytest
from requests import HTTPError
from src.ingestion_service.document_management_client import DocumentManagementClient
from src.ingestion_service.document_ingestor import DocumentIngestor
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import VectorStoreBuilder
from src.shared.constants import DocumentStatus
from src.shared.exceptions import NoDocumentsException


class TestDocumentIngestor:
    @fixture
    def mock_dms_client(self):
        return Mock(spec=DocumentManagementClient)

    @fixture
    def mock_vector_store_builder(self):
        return Mock(spec=VectorStoreBuilder)

    @fixture
    def mock_file_loader(self):
        return Mock(spec=FileLoader)

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_document_skips_completed_document(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        document = "document_already_in_DMS"
        mock_dms_client.get_document_status.return_value = DocumentStatus.COMPLETED
        doc_ingestor.ingest_document(document)

        mock_dms_client.get_document_status.assert_called_once()
        mock_dms_client.update_document_status.assert_not_called()
        mock_process_document.assert_not_called()
        mock_vector_store_builder.add_documents_to_vector_store.assert_not_called()

    @pytest.mark.parametrize(
        "status", [DocumentStatus.PENDING, DocumentStatus.ERROR, None]
    )
    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_document_processes_document(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
        status,
    ):
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        document = f"Document in {status} status"
        mock_dms_client.get_document_status.return_value = status
        doc_ingestor.ingest_document(document)

        mock_dms_client.get_document_status.assert_called_once()
        assert mock_dms_client.update_document_status.call_count == 2
        mock_dms_client.update_document_status.assert_has_calls(
            [
                call(ANY, DocumentStatus.PENDING, ANY),
                call(ANY, DocumentStatus.COMPLETED, ANY),
            ]
        )
        mock_process_document.assert_called_once()
        mock_vector_store_builder.add_documents_to_vector_store.assert_called_once()

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_document_process_document_error(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        document = "Document in PENDING status"
        mock_dms_client.get_document_status.return_value = DocumentStatus.PENDING
        mock_process_document.return_value = None  # FileNotFound

        with pytest.raises(NoDocumentsException):
            doc_ingestor.ingest_document(document)

        mock_dms_client.get_document_status.assert_called_once()
        assert mock_dms_client.update_document_status.call_count == 2
        mock_dms_client.update_document_status.assert_has_calls(
            [
                call(ANY, DocumentStatus.PENDING, ANY),
                call(ANY, DocumentStatus.ERROR, ANY),
            ]
        )
        mock_process_document.assert_called_once()
        mock_vector_store_builder.add_documents_to_vector_store.assert_not_called()

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_document_vector_store_error(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        document = "Document in PENDING status"
        mock_dms_client.get_document_status.return_value = DocumentStatus.PENDING
        mock_vector_store_builder.add_documents_to_vector_store.side_effect = (
            RuntimeError("Runtime error")
        )

        with pytest.raises(RuntimeError):
            doc_ingestor.ingest_document(document)

        mock_dms_client.get_document_status.assert_called_once()
        assert mock_dms_client.update_document_status.call_count == 2
        mock_dms_client.update_document_status.assert_has_calls(
            [
                call(ANY, DocumentStatus.PENDING, ANY),
                call(ANY, DocumentStatus.ERROR, ANY),
            ]
        )
        mock_process_document.assert_called_once()
        mock_vector_store_builder.add_documents_to_vector_store.assert_called_once()

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_document_dms_error_updating_status(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        document = "Document in PENDING status"
        mock_dms_client.get_document_status.return_value = DocumentStatus.PENDING
        mock_dms_client.update_document_status.side_effect = HTTPError("DMS exception")

        with pytest.raises(HTTPError):
            doc_ingestor.ingest_document(document)

        mock_dms_client.get_document_status.assert_called_once()
        mock_dms_client.update_document_status.assert_has_calls(
            [
                call(ANY, DocumentStatus.PENDING, ANY),
                call(ANY, DocumentStatus.ERROR, ANY),  # Called by _try_set_error_status
            ]
        )
        mock_process_document.assert_not_called()
        mock_vector_store_builder.add_documents_to_vector_store.assert_not_called()

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_document_dms_error_getting_status(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        document = "Document in PENDING status"
        mock_dms_client.get_document_status.side_effect = HTTPError("DMS exception")

        with pytest.raises(HTTPError):
            doc_ingestor.ingest_document(document)

        mock_dms_client.get_document_status.assert_called_once()
        mock_dms_client.update_document_status.assert_not_called()
        mock_process_document.assert_not_called()
        mock_vector_store_builder.add_documents_to_vector_store.assert_not_called()

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_documents_empty_list(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        documents = []
        doc_ingestor.ingest_documents(documents)

        mock_dms_client.get_document_status.assert_not_called()
        mock_dms_client.update_document_status.assert_not_called()
        mock_process_document.assert_not_called()
        mock_vector_store_builder.add_documents_to_vector_store.assert_not_called()

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_documents_completed_document(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        documents = ["completed_document"]
        mock_dms_client.get_document_status.return_value = DocumentStatus.COMPLETED

        doc_ingestor.ingest_documents(documents)

        mock_dms_client.get_document_status.assert_called_once()
        mock_dms_client.update_document_status.assert_not_called()
        mock_process_document.assert_not_called()
        mock_vector_store_builder.add_documents_to_vector_store.assert_not_called()

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_documents_new_document(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        documents = ["new_document"]
        mock_dms_client.get_document_status.return_value = None

        doc_ingestor.ingest_documents(documents)

        mock_dms_client.get_document_status.assert_called_once()
        assert mock_dms_client.update_document_status.call_count == 2
        mock_dms_client.update_document_status.assert_has_calls(
            [
                call(ANY, DocumentStatus.PENDING, ANY),
                call(ANY, DocumentStatus.COMPLETED, ANY),
            ]
        )
        mock_process_document.assert_called_once()
        mock_vector_store_builder.add_documents_to_vector_store.assert_called_once()

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_documents_only_process_one_document(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        documents = ["new_document", "completed_document"]
        mock_dms_client.get_document_status.side_effect = [
            None,
            DocumentStatus.COMPLETED,
        ]

        doc_ingestor.ingest_documents(documents)

        assert mock_dms_client.get_document_status.call_count == 2
        assert mock_dms_client.update_document_status.call_count == 2
        mock_dms_client.update_document_status.assert_has_calls(
            [
                call(ANY, DocumentStatus.PENDING, ANY),
                call(ANY, DocumentStatus.COMPLETED, ANY),
            ]
        )
        mock_process_document.assert_called_once()
        mock_vector_store_builder.add_documents_to_vector_store.assert_called_once()

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_documents_one_doc_error_rest_processed(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        documents = ["new_document", "vector_store_error", "new_document2"]
        mock_dms_client.get_document_status.return_value = None
        mock_vector_store_builder.add_documents_to_vector_store.side_effect = [
            Mock(),
            RuntimeError("Runtime error"),
            Mock(),
        ]

        doc_ingestor.ingest_documents(documents)

        assert mock_dms_client.get_document_status.call_count == 3
        assert mock_dms_client.update_document_status.call_count == 6
        mock_dms_client.update_document_status.assert_has_calls(
            [
                call(ANY, DocumentStatus.PENDING, ANY),
                call(ANY, DocumentStatus.COMPLETED, ANY),
                call(ANY, DocumentStatus.PENDING, ANY),
                call(ANY, DocumentStatus.ERROR, ANY),
                call(ANY, DocumentStatus.PENDING, ANY),
                call(ANY, DocumentStatus.COMPLETED, ANY),
            ]
        )
        assert mock_process_document.call_count == 3
        assert mock_vector_store_builder.add_documents_to_vector_store.call_count == 3
