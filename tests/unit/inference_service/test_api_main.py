from contextlib import asynccontextmanager, contextmanager
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.inference_service import main as api_main
from src.shared.constants import DocumentStatus


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


def test_health_check_one_document():
    vector_store_loader = Mock()
    vector_store_loader.get_collection_count.return_value = 1
    api_main.app.state.vector_store_loader = vector_store_loader

    dms_client = Mock()
    dms_response = [
        {
            "doc_hash": "Doc Hash 1",
            "doc_name": "Doc Name 1",
            "status": DocumentStatus.PENDING,
        },
    ]
    dms_client.get_documents.return_value = dms_response
    api_main.app.state.dms_client = dms_client

    with _build_client_no_lifespan() as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "documents_loaded_in_vector_store": "1",
            "documents_loaded_in_dms": f"{dms_response}",
        }


def test_health_check_two_documents():
    vector_store_loader = Mock()
    vector_store_loader.get_collection_count.return_value = 2
    api_main.app.state.vector_store_loader = vector_store_loader

    dms_client = Mock()
    dms_response = [
        {
            "doc_hash": "Doc Hash 1",
            "doc_name": "Doc Name 1",
            "status": DocumentStatus.PENDING,
        },
        {
            "doc_hash": "Doc Hash 2",
            "doc_name": "Doc Name 2",
            "status": DocumentStatus.ERROR,
        },
    ]
    dms_client.get_documents.return_value = dms_response
    api_main.app.state.dms_client = dms_client

    with _build_client_no_lifespan() as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "documents_loaded_in_vector_store": "2",
            "documents_loaded_in_dms": f"{dms_response}",
        }


def test_health_check_no_documents():
    vector_store_loader = Mock()
    vector_store_loader.get_collection_count.return_value = 0
    api_main.app.state.vector_store_loader = vector_store_loader

    dms_client = Mock()
    dms_response = []
    dms_client.get_documents.return_value = dms_response
    api_main.app.state.dms_client = dms_client

    with _build_client_no_lifespan() as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "documents_loaded_in_vector_store": "0",
            "documents_loaded_in_dms": f"{dms_response}",
        }


def test_domain_expert_chat_endpoint():
    session_manager = Mock()
    session = Mock()
    session.session_id = "session-1"
    session.domain_expert_core.ask_question.return_value = "answer"
    session_manager.get_domain_expert_session.return_value = (session, None)
    api_main.app.state.session_manager = session_manager
    vector_store_loader = Mock()
    vector_store_loader.get_collection_count.return_value = 5
    api_main.app.state.vector_store_loader = vector_store_loader

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
    exam_prep_core = Mock()
    exam_prep_core.get_question.return_value = "generated question"
    api_main.app.state.exam_prep_core = exam_prep_core

    with _build_client_no_lifespan() as client:
        response = client.post(
            "/chat/exam-prep/get_question/", json={"user_topic": "Vectors"}
        )

        assert response.status_code == 200
        assert response.json() == {"llm_question": "generated question"}
        exam_prep_core.get_question.assert_called_once_with("Vectors")


def test_exam_prep_get_feedback_endpoint():
    exam_prep_core = Mock()
    exam_prep_core.get_feedback.return_value = "feedback"
    api_main.app.state.exam_prep_core = exam_prep_core

    with _build_client_no_lifespan() as client:
        response = client.post(
            "/chat/exam-prep/get_feedback/",
            json={
                "llm_question": "What is Chroma?",
                "user_answer": "Vector store library",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"feedback": "feedback"}
        exam_prep_core.get_feedback.assert_called_once_with(
            "What is Chroma?", "Vector store library"
        )


def test_domain_expert_request_validation_error():
    vector_store_loader = Mock()
    vector_store_loader.get_collection_count.return_value = 5
    api_main.app.state.vector_store_loader = vector_store_loader

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
