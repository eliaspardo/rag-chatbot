from typing import Union
import logging
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference_service.lifespan import lifespan

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)


class DomainExpertRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Union[None, str] = None


class DomainExpertResponse(BaseModel):
    answer: str
    session_id: str
    system_message: Union[str, None] = None


class ExamPrepQuestionRequest(BaseModel):
    user_topic: str = Field(..., min_length=1)


class ExamPrepQuestionResponse(BaseModel):
    llm_question: str


class ExamPrepFeedbackRequest(BaseModel):
    llm_question: str = Field(..., min_length=1)
    user_answer: str = Field(..., min_length=1)


class ExamPrepFeedbackResponse(BaseModel):
    feedback: str


def get_vectordb_collection_count() -> int:
    return app.state.vector_store_loader.get_collection_count()


def ensure_vector_store_ready():
    if get_vectordb_collection_count() == 0:
        raise HTTPException(503, "Vector store not ready")


@app.get("/health")
def health():
    return {"status": "ok", "documents_loaded": f"{get_vectordb_collection_count()}"}


@app.post(
    "/chat/domain-expert/",
    response_model=DomainExpertResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(ensure_vector_store_ready)],
)
def ask_question(request: DomainExpertRequest):
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


@app.post(
    "/chat/exam-prep/get_question/",
    response_model=ExamPrepQuestionResponse,
    response_model_exclude_none=True,
)
def get_question(question_request: ExamPrepQuestionRequest):
    try:
        llm_question = app.state.exam_prep_core.get_question(
            question_request.user_topic
        )
        return ExamPrepQuestionResponse(
            llm_question=llm_question,
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Processing failed")


@app.post(
    "/chat/exam-prep/get_feedback/",
    response_model=ExamPrepFeedbackResponse,
    response_model_exclude_none=True,
)
def get_feedback(feedback_request: ExamPrepFeedbackRequest):
    try:
        feedback = app.state.exam_prep_core.get_feedback(
            feedback_request.llm_question, feedback_request.user_answer
        )
        return ExamPrepFeedbackResponse(
            feedback=feedback,
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Processing failed")
