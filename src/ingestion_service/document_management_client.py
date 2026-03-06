from typing import List
import requests
from src.shared.models import (
    GetDocumentStatusResponse,
    SetDocumentStatusRequest,
    DMSDocument,
)
from src.shared.constants import DocumentStatus


class DocumentManagementClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_document_status(self, doc_hash: str) -> DocumentStatus | None:
        try:
            response = requests.get(f"{self.base_url}/documents/{doc_hash}/status")
            if response.status_code == 404:
                return None
        except Exception:
            raise
        response.raise_for_status()
        parsed_response = GetDocumentStatusResponse(**response.json())
        return DocumentStatus(parsed_response.status)

    def update_document_status(
        self, doc_hash: str, document_status: DocumentStatus
    ) -> None:
        try:
            request_body = SetDocumentStatusRequest(status=document_status)
            response = requests.put(
                f"{self.base_url}/documents/{doc_hash}/status",
                json=request_body.model_dump(),
            )
        except Exception:
            raise
        response.raise_for_status()

    def get_documents(self) -> List[DMSDocument]:
        try:
            response = requests.get(f"{self.base_url}/documents")
            if response.status_code == 204:
                return []
        except Exception:
            raise
        response.raise_for_status()
        return [DMSDocument(**item) for item in response.json()]
