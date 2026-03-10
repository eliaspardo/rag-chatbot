from pydantic import Field
from pydantic import BaseModel
from src.shared.constants import DocumentStatus


class DMSDocument(BaseModel):
    doc_hash: str = Field(..., min_length=1)
    doc_name: str = Field(..., min_length=1)
    status: DocumentStatus


class GetDocumentStatusResponse(BaseModel):
    doc_name: str = Field(..., min_length=1)
    status: DocumentStatus


class SetDocumentStatusRequest(BaseModel):
    doc_name: str = Field(..., min_length=1)
    status: DocumentStatus
