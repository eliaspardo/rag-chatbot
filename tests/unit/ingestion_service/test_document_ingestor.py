from unittest.mock import ANY, Mock, call, patch
from pytest import fixture, mark
import pytest
from requests import HTTPError
from src.ingestion_service.document_management_client import DocumentManagementClient
from src.ingestion_service.document_ingestor import DocumentIngestor
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import VectorStoreBuilder
from src.shared.constants import DocumentStatus
from src.shared.exceptions import DocumentHashConflictException, NoDocumentsException


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
                call(ANY, ANY, DocumentStatus.PENDING),
                call(ANY, ANY, DocumentStatus.COMPLETED),
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
                call(ANY, ANY, DocumentStatus.PENDING),
                call(ANY, ANY, DocumentStatus.ERROR),
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
                call(ANY, ANY, DocumentStatus.PENDING),
                call(ANY, ANY, DocumentStatus.ERROR),
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
                call(ANY, ANY, DocumentStatus.PENDING),
                call(ANY, ANY, DocumentStatus.ERROR),  # Called by _try_set_error_status
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
                call(ANY, ANY, DocumentStatus.PENDING),
                call(ANY, ANY, DocumentStatus.COMPLETED),
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
                call(ANY, ANY, DocumentStatus.PENDING),
                call(ANY, ANY, DocumentStatus.COMPLETED),
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
                call(ANY, ANY, DocumentStatus.PENDING),
                call(ANY, ANY, DocumentStatus.COMPLETED),
                call(ANY, ANY, DocumentStatus.PENDING),
                call(ANY, ANY, DocumentStatus.ERROR),
                call(ANY, ANY, DocumentStatus.PENDING),
                call(ANY, ANY, DocumentStatus.COMPLETED),
            ]
        )
        assert mock_process_document.call_count == 3
        assert mock_vector_store_builder.add_documents_to_vector_store.call_count == 3

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_document_hash_conflict_does_not_set_error_status(
        self,
        mock_process_document,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        """When DocumentHashConflictException is raised, do not set ERROR status."""
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )
        document = "Document"
        mock_dms_client.update_document_status.side_effect = (
            DocumentHashConflictException()
        )

        with pytest.raises(DocumentHashConflictException):
            doc_ingestor.ingest_document(document)

        mock_dms_client.get_document_status.assert_called_once()
        # Only called once (PENDING) - no ERROR status should be set
        mock_dms_client.update_document_status.assert_called_once_with(
            ANY, ANY, DocumentStatus.PENDING
        )
        # Processing should not happen
        mock_process_document.assert_not_called()
        mock_vector_store_builder.add_documents_to_vector_store.assert_not_called()

    @mark.parametrize(
        "document_path,expected_name",
        [
            # Local paths
            ("/local/path/to/report.pdf", "report.pdf"),
            ("/home/user/documents/analysis.pdf", "analysis.pdf"),
            ("simple.pdf", "simple.pdf"),
            # S3 URIs
            ("s3://bucket/folder/document.pdf", "document.pdf"),
            ("s3://my-bucket/path/to/analysis.pdf", "analysis.pdf"),
            # HTTPS URLs
            ("https://bucket.s3.amazonaws.com/docs/summary.pdf", "summary.pdf"),
            ("https://example.com/files/report.pdf", "report.pdf"),
        ],
    )
    def test_extract_doc_name_from_various_paths(
        self,
        mock_file_loader,
        mock_vector_store_builder,
        mock_dms_client,
        document_path,
        expected_name,
    ):
        """Test that doc_name is correctly extracted from various path formats."""
        doc_ingestor = DocumentIngestor(
            mock_dms_client, mock_vector_store_builder, mock_file_loader, print
        )

        result = doc_ingestor._extract_doc_name(document_path)

        assert result == expected_name
