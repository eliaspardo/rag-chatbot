"""HTTP client for the inference service, used by the Streamlit UI."""

import logging
from dataclasses import dataclass, field
import os
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 5
CHAT_TIMEOUT = int(os.getenv("CHAT_TIMEOUT", "120"))


class NoDocumentsIngestedError(Exception):
    """Raised when the inference service reports no documents have been ingested."""


@dataclass
class DocumentInfo:
    """Lightweight representation of a document entry returned by the health endpoint."""

    doc_hash: str
    doc_name: str
    status: str


@dataclass
class HealthStatus:
    """Health status of the inference service including vector store and document info."""

    is_healthy: bool
    vector_store_count: int = 0
    documents: List[DocumentInfo] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class ChatResponse:
    """Response from the domain expert chat endpoint."""

    answer: str
    session_id: str
    system_message: Optional[str] = None


class InferenceServiceClient:
    """Client for interacting with the inference service REST API."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_health(self) -> HealthStatus:
        """Call the inference service health endpoint and return a structured HealthStatus."""
        try:
            response = requests.get(
                f"{self.base_url}/health", timeout=HEALTH_CHECK_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            documents = [
                DocumentInfo(
                    doc_hash=doc["doc_hash"],
                    doc_name=doc["doc_name"],
                    status=doc["status"],
                )
                for doc in data.get("documents_loaded_in_dms", [])
            ]
            return HealthStatus(
                is_healthy=True,
                # The API returns documents_loaded_in_vector_store as a string; cast to int.
                vector_store_count=int(data.get("documents_loaded_in_vector_store", 0)),
                documents=documents,
            )
        except requests.Timeout:
            return HealthStatus(is_healthy=False, error_message="Health check timeout")
        except requests.ConnectionError:
            return HealthStatus(
                is_healthy=False, error_message="Inference Service: unreachable"
            )
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.error("Health check failed: %s", exc)
            return HealthStatus(is_healthy=False, error_message=str(exc))

    def ask_question(
        self, question: str, session_id: Optional[str] = None
    ) -> ChatResponse:
        """Post a question to the domain expert endpoint and return the ChatResponse."""
        response = requests.post(
            f"{self.base_url}/chat/domain-expert/",
            json={"question": question, "session_id": session_id},
            timeout=CHAT_TIMEOUT,
        )
        if response.status_code == 503:
            try:
                detail = response.json().get("detail", "Service unavailable.")
            except ValueError:
                detail = "Service unavailable."
            if (
                "no documents" in detail.lower()
                or "not been ingested" in detail.lower()
            ):
                raise NoDocumentsIngestedError(detail)
            response.raise_for_status()
        response.raise_for_status()
        data = response.json()
        return ChatResponse(
            answer=data["answer"],
            session_id=data["session_id"],
            system_message=data.get("system_message"),
        )
