from unittest.mock import Mock, patch
from pytest import fixture
from src.ingestion_service.document_management_client import DocumentManagementClient
from src.ingestion_service.dms_ingestion import ingest_documents
from src.ingestion_service.vector_store_builder import VectorStoreBuilder
from src.shared.constants import DocumentStatus

sample_hash = "d41d8cd98f00b204e9800998ecf8427e"


class TestDMSIngestion:
    @fixture
    def mock_dms_client(self):
        return Mock(spec=DocumentManagementClient)

    @fixture
    def mock_vector_store_builder(self):
        return Mock(spec=VectorStoreBuilder)

    @patch("src.ingestion_service.dms_ingestion.process_documents")
    def test_request_ingest_skips_completed_document(
        self,
        mock_process_documents,
        mock_vector_store_builder,
        mock_dms_client,
    ):
        document_list = ["document_already_in_DMS"]
        mock_dms_client.get_document_status.return_value = DocumentStatus.COMPLETED
        ingest_documents(document_list, mock_dms_client, mock_vector_store_builder)

        mock_dms_client.get_document_status.assert_called_once()
        mock_dms_client.update_document_status.assert_not_called()
        mock_process_documents.assert_not_called()
        mock_vector_store_builder.add_documents_to_vector_store.assert_not_called()
