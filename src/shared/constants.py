from enum import Enum


class Error(Enum):
    NO_DOCUMENTS = "No documents provided"
    EXCEPTION = "Exception"


class DocumentStatus(str, Enum):
    INITIALIZED = "Document registered"
    PENDING = "Document pending processing"
    COMPLETED = "Document processing completed"
