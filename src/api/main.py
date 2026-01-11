from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.lifespan import lifespan

app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Union[None, str] = None


class Reply(BaseModel):
    answer: str = Field(..., min_length=1)
    session_id: str


class Topic(BaseModel):
    topic: str = Field(..., min_length=1)


@app.get("/health")
def read_root():
    return {"status": "ok"}


@app.post("/chat/domain-expert/")
def ask_question(query: Query):
    if not query.question or query.question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    domain_expert_session = app.state.session_manager.get_domain_expert_session(
        query.session_id
    )
    answer = domain_expert_session.domain_expert_core.ask_question(query.question)
    session_id = domain_expert_session.id
    return Reply(answer=answer, session_id=session_id)


@app.post("/chat/exam-prep/question")
def get_question(topic: Query):
    if not topic.question or topic.question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    exam_prep = app.state.exam_prep
    question = exam_prep.ask_question(topic.question)
    return question


@app.post("/chat/exam-prep/answer")
def get_answer(answer: Query):
    if not answer.question or answer.question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    exam_prep = app.state.exam_prep
    answer = exam_prep.get_answer(answer.question)
    return answer
