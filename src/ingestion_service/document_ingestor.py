import os
import hashlib
from typing import List
from src.ingestion_service.bootstrap import ProgressCallback, process_document
from src.ingestion_service.document_management_client import DocumentManagementClient
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import VectorStoreBuilder
import logging
from src.shared.constants import DocumentStatus
from src.shared.env_loader import load_environment
from src.shared.exceptions import IngestionRequestException, NoDocumentsException

logger = logging.getLogger(__name__)

load_environment()
DMS_URL = os.getenv("DMS_URL")


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
            self.ingest_document(document)

    def ingest_document(
        self,
        document: str,
    ) -> None:
        doc_hash = hashlib.md5(document.encode()).hexdigest()
        doc_status = self.dms_client.get_document_status(doc_hash, DMS_URL)
        if doc_status is None or doc_status != DocumentStatus.COMPLETED:
            self.dms_client.update_document_status(
                doc_hash, DocumentStatus.PENDING, DMS_URL
            )
            try:
                docs = process_document(
                    document, self.file_loader, self.vector_store_builder, self.progress
                )
            except NoDocumentsException:
                logger.error(
                    "Error seeding vector store: no documents found after splitting!"
                )
                # TODO - Signal it's ERROR
                raise NoDocumentsException()
            if docs:
                self.progress(f"🏭 Adding docs from {document} to vector store.")
                self.vector_store_builder.add_documents_to_vector_store(docs)
                self.progress(f"✅ Docs from {document} saved.")
                self.dms_client.update_document_status(
                    doc_hash, DocumentStatus.COMPLETED, DMS_URL
                )
