import hashlib
from typing import List
from src.ingestion_service.bootstrap import ProgressCallback, process_document
from src.ingestion_service.document_management_client import DocumentManagementClient
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import VectorStoreBuilder
import logging
from src.shared.constants import DocumentStatus
from src.shared.exceptions import (
    IngestionRequestException,
    NoDocumentsException,
)


logger = logging.getLogger(__name__)


class DocumentIngestor:
    def __init__(
        self,
        dms_client: DocumentManagementClient,
        vector_store_builder: VectorStoreBuilder,
        file_loader: FileLoader,
        progress: ProgressCallback,
    ):
        self.dms_client = dms_client
        self.vector_store_builder = vector_store_builder
        self.file_loader = file_loader
        self.progress = progress

    def ingest_documents(
        self,
        doc_list: List[str],
    ):
        try:
            clean_pdf_paths = [p.strip() for p in doc_list if p.strip()]
        except Exception:
            raise IngestionRequestException("Error when reading PDFs provided.")
        for document in clean_pdf_paths:
            try:
                self.ingest_document(document)
            except Exception:
                logger.error(f"Could not ingest {document}.")

    def ingest_document(
        self,
        document: str,
    ) -> None:
        doc_hash = hashlib.md5(document.encode()).hexdigest()
        try:
            doc_status = self.dms_client.get_document_status(doc_hash)
        except Exception:
            logger.error(f"Could not get status for {document}, skipping processing")
            raise
        if doc_status != DocumentStatus.COMPLETED:
            try:
                self.dms_client.update_document_status(doc_hash, DocumentStatus.PENDING)
                docs = process_document(
                    document, self.file_loader, self.vector_store_builder, self.progress
                )
                if not docs:
                    logger.error(f"Error processing {document}: No documents!")
                    raise NoDocumentsException()
                else:
                    self.progress(f"🏭 Adding docs from {document} to vector store.")
                    self.vector_store_builder.add_documents_to_vector_store(docs)
                    self.progress(f"✅ Docs from {document} saved.")
                    self.dms_client.update_document_status(
                        doc_hash, DocumentStatus.COMPLETED
                    )
            except Exception as e:
                logger.error(f"Failed to ingest {document}: {e}")
                self._try_set_error_status(doc_hash, document)
                raise

    # To be called when there's an exception processing
    def _try_set_error_status(self, doc_hash: str, document: str):
        try:
            self.dms_client.update_document_status(doc_hash, DocumentStatus.ERROR)
        except Exception:
            logger.warning(f"Could not set ERROR status for {document}")
