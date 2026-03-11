from enum import Enum


class Error(Enum):
    NO_DOCUMENTS = "No documents provided"
    EXCEPTION = "Exception"


class DocumentStatus(str, Enum):
    PENDING = "Document pending processing"
    COMPLETED = "Document processing completed"
    ERROR = "Error in document processing"


class SetDocumentResult(Enum):
    CREATED = "created"
    UPDATED = "updated"
