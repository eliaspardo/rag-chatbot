from src.shared.constants import DocumentStatus


class DBClient:
    def get_document_name(self, doc_hash):
        return "a"

    def get_document_status(self, doc_hash) -> DocumentStatus:
        return DocumentStatus.ERROR
