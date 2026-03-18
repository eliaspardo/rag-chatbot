import logging
from dataclasses import dataclass, field
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30


@dataclass
class DocumentInfo:
    doc_hash: str
    doc_name: str
    status: str


@dataclass
class HealthStatus:
    is_healthy: bool
    vector_store_count: int = 0
    documents: List[DocumentInfo] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class ChatResponse:
    answer: str
    session_id: str
    system_message: Optional[str] = None


class InferenceServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_health(self) -> HealthStatus:
        try:
            response = requests.get(
                f"{self.base_url}/health", timeout=HEALTH_CHECK_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            data = response.json()
            documents = [
                DocumentInfo(**doc) for doc in data.get("documents_loaded_in_dms", [])
            ]
            return HealthStatus(
                is_healthy=True,
                vector_store_count=int(
                    data.get("documents_loaded_in_vector_store", 0)
                ),
                documents=documents,
            )
        except requests.Timeout:
            return HealthStatus(is_healthy=False, error_message="Health check timeout")
        except requests.ConnectionError:
            return HealthStatus(
                is_healthy=False, error_message="Inference Service: unreachable"
            )
        except Exception as e:
            logger.error(e)
            return HealthStatus(is_healthy=False, error_message=str(e))

    def ask_question(
        self, question: str, session_id: Optional[str] = None
    ) -> ChatResponse:
        response = requests.post(
            f"{self.base_url}/chat/domain-expert/",
            json={"question": question, "session_id": session_id},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return ChatResponse(
            answer=data["answer"],
            session_id=data["session_id"],
            system_message=data.get("system_message"),
        )
