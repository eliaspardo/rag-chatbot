from typing import List
from src.shared.constants import DocumentStatus, SetDocumentResult
from src.shared.models import DMSDocument


class DBClient:
    def get_document_name(self, doc_hash) -> str:
        # Return document name from hash
        return

    def get_document_status(self, doc_hash) -> DocumentStatus:
        # Return document status from hash
        return

    def get_documents(self) -> List[DMSDocument]:
        # Return document list
        return

    def set_document_status(
        self, doc_hash, doc_name, status
    ) -> tuple[DMSDocument, SetDocumentResult]:
        # Return document and operation result
        return
        (
            DMSDocument(
                doc_hash="afsdaf", doc_name="blabl", status=DocumentStatus.PENDING
            ),
            SetDocumentResult.CREATED,
        )
