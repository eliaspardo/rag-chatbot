from fastapi.testclient import TestClient
import pytest
import chromadb
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.document_management_service.models import DBDMSDocument
from playwright.sync_api import Page, expect

from src.ingestion_service.main import SingleIngestionRequest


CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")
DMS_DATABASE_URL = "postgresql://dms:dms@localhost:5432/dms"
INGESTION_URL = "http://127.0.0.1:8003/ingestion/documents/"
document_path = "https://istqb-documents.s3.eu-central-003.backblazeb2.com/ISTQB_CTAL-TM_Syllabus_v3.0.pdf"
document_request = SingleIngestionRequest(document=document_path)
ingested_document_string = (
    "ISTQB_CTAL-TM_Syllabus_v3.0.pdf — Document processing completed"
)


class TestE2EFlow:
    @pytest.fixture()
    def clear_vector_db(self):
        """Fixture that clears the ChromaDB collection"""
        try:
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            collection = chroma_client.get_collection(CHROMA_COLLECTION)
            # Get all document IDs
            all_ids = collection.get()["ids"]
            # Delete all documents
            if all_ids:
                collection.delete(ids=all_ids)
            yield
            # Clean up again after test
            collection = chroma_client.get_collection(CHROMA_COLLECTION)
            # Get all document IDs
            all_ids = collection.get()["ids"]
            # Delete all documents
            if all_ids:
                collection.delete(ids=all_ids)
        except Exception:
            pass

    @pytest.fixture()
    def clear_dms_db(self):
        """Fixture that clears all document records from DMS"""
        engine = create_engine(DMS_DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            session.query(DBDMSDocument).delete()
            session.commit()
            yield
            # Clean up again after test
            session.query(DBDMSDocument).delete()
            session.commit()
        finally:
            session.close()
            engine.dispose()

    @pytest.fixture
    def ingestion_client(self):
        """Create FastAPI TestClient."""

        from src.ingestion_service.main import app

        with TestClient(app) as test_client:
            yield test_client

    def test_e2e_flow(
        self, clear_vector_db, clear_dms_db, ingestion_client, page: Page
    ):
        # Arrange
        inference_string = "How can I calculate the total cost of quality?"
        inference_response_string = "Defect Prevention Costs"
        chat_input = page.get_by_role("textbox")
        chat_submit_button = page.get_by_test_id("stChatInputSubmitButton")
        alert_container = page.get_by_test_id("stAlertContentError")
        system_status_sidebar_button = page.get_by_text("System Status")
        chat_sidebar_button = page.get_by_text("Chat")
        vector_store_doc_count = page.get_by_test_id("documents_in_vector_store_count")
        refresh_button = page.get_by_role("button", name="Refresh")
        ingested_document_item = page.get_by_text(ingested_document_string)
        inference_response_item = page.get_by_test_id("stChatMessage").nth(2)
        chat_thinking_item = page.get_by_text("Thinking")

        # Act - Navigate to chat interface and send inference request
        page.goto("http://localhost:8501")
        chat_input.type(inference_string)
        chat_submit_button.click()
        # Assert - Verify error message is displayed
        expect(alert_container).to_have_text(
            ("Request failed: 503 Server Error: Service Unavailable for url: ")
            + ("http://inference_service:8000/chat/domain-expert/")
        )
        # Act - Navigate to system status sidebar
        system_status_sidebar_button.click()
        # Assert - Verify vector store document count is 0
        expect(vector_store_doc_count).to_have_text("0")
        # Act - Ingest document and click refresh button
        ingestion_client.post(
            "/ingestion/document/", json=document_request.model_dump()
        )
        refresh_button.click()
        # Assert - Verify vector store document count and ingested document item is visible
        expect(vector_store_doc_count).to_have_text("197")
        expect(ingested_document_item).to_be_visible()
        # Act - Navigate to chat sidebar and send inference request
        chat_sidebar_button.click()
        chat_input.type(inference_string)
        chat_submit_button.click()
        # Assert - Verify thinking item is displayed and then disappears, and response is displayed
        expect(chat_thinking_item).to_be_visible()
        expect(chat_thinking_item).not_to_be_visible(timeout=20_000)
        expect(inference_response_item).to_contain_text(
            inference_response_string, ignore_case=True
        )
