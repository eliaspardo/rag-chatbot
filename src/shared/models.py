"""Shared Pydantic models used across multiple services."""

from pydantic import Field
from pydantic import BaseModel
from src.shared.constants import DocumentStatus


class DMSDocument(BaseModel):
    """Pydantic representation of a document record in the Document Management Service."""

    doc_hash: str = Field(..., min_length=1)
    doc_name: str = Field(..., min_length=1)
    status: DocumentStatus


class GetDocumentStatusResponse(BaseModel):
    """Response schema for the GET document status endpoint."""

    doc_name: str = Field(..., min_length=1)
    status: DocumentStatus


class SetDocumentStatusRequest(BaseModel):
    """Request schema for the PUT document status endpoint."""

    doc_name: str = Field(..., min_length=1)
    status: DocumentStatus
