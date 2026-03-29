"""Document ingestor orchestrating DMS registration and vector store population."""

import hashlib
import os
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import List
from src.ingestion_service.bootstrap import ProgressCallback, process_document
from src.ingestion_service.document_management_client import DocumentManagementClient
from src.ingestion_service.file_loader import FileLoader
from src.ingestion_service.vector_store_builder import VectorStoreBuilder
import logging
from src.shared.constants import DocumentStatus
from src.shared.exceptions import (
    DocumentHashConflictException,
    IngestionRequestException,
    NoDocumentsException,
)


logger = logging.getLogger(__name__)


@dataclass
class DocumentIngestionResult:
    """Result of a single document ingestion attempt."""

    document: str
    success: bool
    error: str | None = None


class DocumentIngestor:
    """Orchestrates end-to-end ingestion: DMS status tracking and vector store loading."""

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
    ) -> List[DocumentIngestionResult]:
        """Ingest multiple documents, returning per-document success or failure results."""
        try:
            clean_pdf_paths = [p.strip() for p in doc_list if p.strip()]
        except Exception:
            raise IngestionRequestException("Error when reading PDFs provided.")
        results = []
        for document in clean_pdf_paths:
            try:
                self.ingest_document(document)
                results.append(DocumentIngestionResult(document=document, success=True))
            except Exception as e:
                logger.error(f"Could not ingest {document}.")
                logger.exception(e)
                results.append(
                    DocumentIngestionResult(
                        document=document, success=False, error=str(e)
                    )
                )
        return results

    def ingest_document(
        self,
        document: str,
    ) -> None:
        """Ingest a single document into the vector store, skipping already-completed ones."""
        doc_hash = hashlib.md5(document.encode()).hexdigest()
        doc_name = self._extract_doc_name(document)
        try:
            doc_status = self.dms_client.get_document_status(doc_hash)
        except Exception:
            logger.error(f"Could not get status for {document}, skipping processing")
            raise
        if doc_status != DocumentStatus.COMPLETED:
            try:
                self.dms_client.update_document_status(
                    doc_hash, doc_name, DocumentStatus.PENDING
                )
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
                        doc_hash, doc_name, DocumentStatus.COMPLETED
                    )
            except DocumentHashConflictException as hash_exception:
                logger.error(f"Failed to ingest {document}: {hash_exception}")
                raise
            except Exception as e:
                logger.error(f"Failed to ingest {document}: {e}")
                self._try_set_error_status(doc_hash, doc_name, document)
                raise

    def _try_set_error_status(self, doc_hash: str, doc_name: str, document: str):
        """Attempt to mark a document as ERROR in DMS; log a warning on failure."""
        try:
            self.dms_client.update_document_status(
                doc_hash, doc_name, DocumentStatus.ERROR
            )
        except Exception:
            logger.warning(f"Could not set ERROR status for {document}")

    def _extract_doc_name(self, document: str) -> str:
        """Extract the base filename from a local path or URL."""
        parsed = urlparse(document)
        path = parsed.path if parsed.scheme else document
        return os.path.basename(path)
