from pydantic import Field
from pydantic import BaseModel
from src.shared.constants import DocumentStatus


class DocumentStatusResponse(BaseModel):
    doc_id: str = Field(..., min_length=1)
    doc_hash: str = Field(..., min_length=1)
    status: DocumentStatus
