import requests
from src.shared.models import (
    GetDocumentStatusResponse,
    RegisterDocumentRequest,
    RegisterDocumentResponse,
)
from src.shared.constants import DocumentStatus


def get_document_status(doc_hash: str, base_url: str) -> DocumentStatus | None:
    try:
        response = requests.get(f"{base_url}/documents/{doc_hash}/status")
        if response.status_code == 404:
            return None
    except Exception:
        raise
    response.raise_for_status()
    parsed_response = GetDocumentStatusResponse(**response.json())
    return DocumentStatus(parsed_response.status)


def register_document(
    doc_hash: str, doc_name: str, base_url: str
) -> RegisterDocumentResponse:
    try:
        request_body = RegisterDocumentRequest(doc_name=doc_name)
        print(request_body)
        response = requests.post(
            f"{base_url}/documents/{doc_hash}", json=request_body.model_dump()
        )
    except Exception:
        raise
    response.raise_for_status()
    return RegisterDocumentResponse(**response.json())
