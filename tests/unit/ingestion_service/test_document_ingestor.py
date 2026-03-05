from unittest.mock import ANY, Mock, call, patch
from pytest import fixture
from src.ingestion_service.document_management_client import DocumentManagementClient
from src.ingestion_service.document_ingestor import DocumentIngestor
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import VectorStoreBuilder
from src.shared.constants import DocumentStatus


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

    @patch("src.ingestion_service.document_ingestor.process_document")
    def test_ingest_document_processes_error_document(
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
        mock_dms_client.get_document_status.return_value = DocumentStatus.ERROR
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
