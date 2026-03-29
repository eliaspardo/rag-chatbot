"""Shared constants and enumerations used across multiple services."""

from enum import Enum


class Error(Enum):
    """Generic error message constants."""

    NO_DOCUMENTS = "No documents provided"
    EXCEPTION = "Exception"


class DocumentStatus(str, Enum):
    """Processing lifecycle states for an ingested document."""

    PENDING = "Document pending processing"
    COMPLETED = "Document processing completed"
    ERROR = "Error in document processing"


class SetDocumentResult(Enum):
    """Outcome of a set-document-status operation."""

    CREATED = "created"
    UPDATED = "updated"
