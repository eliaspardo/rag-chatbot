"""FastAPI application for the inference service."""

from typing import List, Union
import logging
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference_service.lifespan import lifespan
from src.shared.models import DMSDocument

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)


class DomainExpertRequest(BaseModel):
    """Request body for the domain expert chat endpoint."""

    question: str = Field(..., min_length=1)
    session_id: Union[None, str] = None


class DomainExpertResponse(BaseModel):
    """Response body for the domain expert chat endpoint."""

    answer: str
    session_id: str
    system_message: Union[str, None] = None


def get_vectordb_collection_count() -> int:
    """Return the number of documents currently stored in the vector store."""
    return app.state.vector_store_loader.get_collection_count()


def get_documents() -> List[DMSDocument]:
    """Fetch all documents registered in the Document Management Service."""
    return app.state.dms_client.get_documents()


def ensure_vector_store_ready():
    """Raise HTTP 503 if the vector store contains no documents."""
    if get_vectordb_collection_count() == 0:
        raise HTTPException(503, "Vector store not ready")


@app.get("/health")
def health():
    """Return service health status including vector store and DMS document counts."""
    documents = [doc.model_dump() for doc in get_documents()]

    return {
        "status": "ok",
        "documents_loaded_in_vector_store": f"{get_vectordb_collection_count()}",
        "documents_loaded_in_dms": documents,
    }


@app.post(
    "/chat/domain-expert/",
    response_model=DomainExpertResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(ensure_vector_store_ready)],
)
def ask_question(request: DomainExpertRequest):
    """Submit a question to the domain expert and return the answer with session context."""
    try:
        (
            domain_expert_session,
            system_message,
        ) = app.state.session_manager.get_domain_expert_session(request.session_id)
        answer = domain_expert_session.domain_expert_core.ask_question(request.question)
        return DomainExpertResponse(
            answer=answer,
            session_id=domain_expert_session.session_id,
            system_message=system_message,
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Processing failed")
