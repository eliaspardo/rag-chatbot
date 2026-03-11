from typing import List
from src.shared.constants import DocumentStatus
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

    def set_document_status(self, doc_hash, doc_name, status) -> DMSDocument:
        # Return document
        return
