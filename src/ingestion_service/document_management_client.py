from typing import List
import requests
from src.shared.models import (
    GetDocumentStatusResponse,
    SetDocumentStatusRequest,
    DMSDocument,
)
from src.shared.constants import DocumentStatus


class DocumentManagementClient:
    def __init__(self, base_url: str, retries: int = 3):
        self.base_url = base_url
        self.retries = retries

    def get_document_status(
        self, doc_hash: str, base_url: str
    ) -> DocumentStatus | None:
        try:
            response = requests.get(f"{base_url}/documents/{doc_hash}/status")
            if response.status_code == 404:
                return None
        except Exception:
            raise
        response.raise_for_status()
        parsed_response = GetDocumentStatusResponse(**response.json())
        return DocumentStatus(parsed_response.status)

    def update_document_status(
        self, doc_hash: str, document_status: DocumentStatus, base_url: str
    ) -> None:
        try:
            request_body = SetDocumentStatusRequest(status=document_status)
            response = requests.put(
                f"{base_url}/documents/{doc_hash}/status",
                json=request_body.model_dump(),
            )
        except Exception:
            raise
        response.raise_for_status()

    def get_documents(self, base_url: str) -> List[DMSDocument]:
        try:
            response = requests.get(f"{base_url}/documents")
            if response.status_code == 204:
                return []
        except Exception:
            raise
        response.raise_for_status()
        return [DMSDocument(**item) for item in response.json()]
