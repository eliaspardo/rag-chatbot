import logging
import sys
from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.lifespan import lifespan

# Force logging to stdout so Render can see it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

logger.info("=== STARTING APPLICATION ===")

app = FastAPI(lifespan=lifespan)

logger.info("=== FastAPI app created ===")


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


def _ensure_app_ready() -> None:
    # If tests inject dependencies without lifespan, treat as ready.
    if hasattr(app.state, "session_manager") and hasattr(app.state, "exam_prep_core"):
        return

    if not getattr(app.state, "bootstrap_ready", True):
        bootstrap_error = getattr(app.state, "bootstrap_error", None)
        if bootstrap_error:
            raise HTTPException(
                status_code=500,
                detail=f"Service bootstrap failed: {bootstrap_error}",
            )
        raise HTTPException(
            status_code=503,
            detail="Service is initializing. Please retry shortly.",
        )


@app.get("/health")
def read_root():
    if not getattr(app.state, "bootstrap_ready", True):
        if getattr(app.state, "bootstrap_error", None):
            return {"status": "error", "ready": False}
        return {"status": "starting", "ready": False}
    return {"status": "ok", "ready": True}


@app.get("/")
def read_index():
    return {"service": "rag-chatbot", "status": "ok"}


@app.post(
    "/chat/domain-expert/",
    response_model=DomainExpertResponse,
    response_model_exclude_none=True,
)
def ask_question(request: DomainExpertRequest):
    _ensure_app_ready()
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


@app.post(
    "/chat/exam-prep/get_question/",
    response_model=ExamPrepQuestionResponse,
    response_model_exclude_none=True,
)
def get_question(question_request: ExamPrepQuestionRequest):
    _ensure_app_ready()
    llm_question = app.state.exam_prep_core.get_question(question_request.user_topic)
    return ExamPrepQuestionResponse(
        llm_question=llm_question,
    )


@app.post(
    "/chat/exam-prep/get_feedback/",
    response_model=ExamPrepFeedbackResponse,
    response_model_exclude_none=True,
)
def get_feedback(feedback_request: ExamPrepFeedbackRequest):
    _ensure_app_ready()
    feedback = app.state.exam_prep_core.get_feedback(
        feedback_request.llm_question, feedback_request.user_answer
    )
    return ExamPrepFeedbackResponse(
        feedback=feedback,
    )
