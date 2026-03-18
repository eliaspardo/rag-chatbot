import logging
from typing import List
import requests
from src.shared.models import DMSDocument

logger = logging.getLogger(__name__)


class DocumentManagementClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_documents(self) -> List[DMSDocument]:
        try:
            response = requests.get(f"{self.base_url}/documents/", timeout=5)
            if response.status_code == 204:
                return []
        except Exception as e:
            logger.error(e)
            raise
        response.raise_for_status()
        return [DMSDocument(**item) for item in response.json()]
