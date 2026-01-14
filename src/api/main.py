from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.api.lifespan import lifespan

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


@app.get("/health")
def read_root():
    return {"status": "ok"}


@app.post(
    "/chat/domain-expert/",
    response_model=DomainExpertResponse,
    response_model_exclude_none=True,
)
def ask_question(request: DomainExpertRequest):
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
    feedback = app.state.exam_prep_core.get_feedback(
        feedback_request.llm_question, feedback_request.user_answer
    )
    return ExamPrepFeedbackResponse(
        feedback=feedback,
    )
