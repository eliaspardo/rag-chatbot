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


class ExamPrepQuestionRequest(BaseModel):
    user_topic: str = Field(..., min_length=1)
    session_id: Union[None, str] = None


class ExamPrepQuestionResponse(BaseModel):
    llm_question: str
    session_id: str


class ExamPrepFeedbackRequest(BaseModel):
    llm_question: str = Field(..., min_length=1)
    user_answer: str = Field(..., min_length=1)
    session_id: Union[None, str] = None


class ExamPrepFeedbackResponse(BaseModel):
    feedback: str
    session_id: str


@app.get("/health")
def read_root():
    return {"status": "ok"}


@app.post("/chat/domain-expert/")
def ask_question(request: DomainExpertRequest):
    domain_expert_session = app.state.session_manager.get_domain_expert_session(
        request.session_id
    )
    answer = domain_expert_session.domain_expert_core.ask_question(request.question)
    return DomainExpertResponse(
        answer=answer, session_id=domain_expert_session.session_id
    )


@app.post("/chat/exam-prep/get_question")
def get_question(question_request: ExamPrepQuestionRequest):
    exam_prep_session = app.state.session_manager.get_exam_prep_session(
        question_request.session_id
    )
    llm_question = exam_prep_session.exam_prep_core.get_question(
        question_request.user_topic
    )
    return ExamPrepQuestionResponse(
        llm_question=llm_question, session_id=exam_prep_session.session_id
    )


@app.post("/chat/exam-prep/get_feedback")
def get_feedback(feedback_request: ExamPrepFeedbackRequest):
    exam_prep_session = app.state.session_manager.get_exam_prep_session(
        feedback_request.session_id
    )
    feedback = exam_prep_session.exam_prep_core.get_feedback(
        feedback_request.llm_question, feedback_request.user_answer
    )
    return ExamPrepFeedbackResponse(
        feedback=feedback, session_id=exam_prep_session.session_id
    )
