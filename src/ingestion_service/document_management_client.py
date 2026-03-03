import requests
from src.shared.models import GetDocumentStatusResponse
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
