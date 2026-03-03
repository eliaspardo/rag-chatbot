import requests
from src.shared.models import DocumentStatusResponse
from src.shared.constants import DocumentStatus


def get_document_status(doc_id: str, base_url: str) -> DocumentStatus | None:
    try:
        response = requests.get(f"{base_url}/documents/{doc_id}/status")
        if response.status_code == 404:
            return None
    except Exception:
        raise Exception
    response.raise_for_status()
    parsed_response = DocumentStatusResponse(**response.json())
    return DocumentStatus(parsed_response.status)
