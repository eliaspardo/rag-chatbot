from contextlib import asynccontextmanager, contextmanager
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.api import main as api_main


@asynccontextmanager
async def _no_lifespan(app):
    yield


@contextmanager
def _build_client_no_lifespan():
    original_lifespan = api_main.app.router.lifespan_context
    api_main.app.router.lifespan_context = _no_lifespan
    try:
        with TestClient(api_main.app) as client:
            yield client
    finally:
        api_main.app.router.lifespan_context = original_lifespan


def test_health_check():
    with _build_client_no_lifespan() as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_domain_expert_chat_endpoint():
    session_manager = Mock()
    session = Mock()
    session.session_id = "session-1"
    session.domain_expert_core.ask_question.return_value = "answer"
    session_manager.get_domain_expert_session.return_value = (session, None)
    api_main.app.state.session_manager = session_manager

    with _build_client_no_lifespan() as client:
        response = client.post(
            "/chat/domain-expert/",
            json={"question": "What is RAG?", "session_id": "existing"},
        )

        assert response.status_code == 200
        assert response.json() == {"answer": "answer", "session_id": "session-1"}
        session_manager.get_domain_expert_session.assert_called_once_with("existing")
        session.domain_expert_core.ask_question.assert_called_once_with("What is RAG?")


def test_exam_prep_get_question_endpoint():
    session_manager = Mock()
    session = Mock()
    session.session_id = "session-2"
    session.exam_prep_core.get_question.return_value = "generated question"
    session_manager.get_exam_prep_session.return_value = (session, "system note")
    api_main.app.state.session_manager = session_manager

    with _build_client_no_lifespan() as client:
        response = client.post(
            "/chat/exam-prep/get_question/",
            json={"user_topic": "Vectors", "session_id": None},
        )

        assert response.status_code == 200
        assert response.json() == {
            "llm_question": "generated question",
            "session_id": "session-2",
            "system_message": "system note",
        }
        session_manager.get_exam_prep_session.assert_called_once_with(None)
        session.exam_prep_core.get_question.assert_called_once_with("Vectors")


def test_exam_prep_get_feedback_endpoint():
    session_manager = Mock()
    session = Mock()
    session.session_id = "session-3"
    session.exam_prep_core.get_feedback.return_value = "feedback"
    session_manager.get_exam_prep_session.return_value = (session, None)
    api_main.app.state.session_manager = session_manager

    with _build_client_no_lifespan() as client:
        response = client.post(
            "/chat/exam-prep/get_feedback/",
            json={
                "llm_question": "What is FAISS?",
                "user_answer": "Vector store library",
                "session_id": "session-3",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"feedback": "feedback", "session_id": "session-3"}
        session_manager.get_exam_prep_session.assert_called_once_with("session-3")
        session.exam_prep_core.get_feedback.assert_called_once_with(
            "What is FAISS?", "Vector store library"
        )


def test_domain_expert_request_validation_error():
    with _build_client_no_lifespan() as client:
        response = client.post("/chat/domain-expert/", json={"question": ""})

        assert response.status_code == 422


def test_exam_prep_question_request_validation_error():
    with _build_client_no_lifespan() as client:
        response = client.post("/chat/exam-prep/get_question/", json={"user_topic": ""})

        assert response.status_code == 422


def test_exam_prep_feedback_request_validation_error():
    with _build_client_no_lifespan() as client:
        response = client.post(
            "/chat/exam-prep/get_feedback/",
            json={"llm_question": "", "user_answer": ""},
        )

        assert response.status_code == 422
