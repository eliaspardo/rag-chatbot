from time import sleep
import pytest
import chromadb
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.document_management_service.models import DBDMSDocument
from playwright.sync_api import Page, expect


CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")
DMS_DATABASE_URL = "postgresql://dms:dms@localhost:5432/dms"


class TestE2EFlow:
    @pytest.fixture()
    def clear_vector_db(self):
        """Fixture that clears the ChromaDB collection"""
        try:
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            chroma_client.delete_collection(CHROMA_COLLECTION)
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

    def test_e2e_flow(self, clear_vector_db, clear_dms_db, page: Page):
        inference_string = "How can I calculate the total cost of quality?"
        chat_input = page.get_by_role("textbox")
        chat_submit_button = page.get_by_test_id("stChatInputSubmitButton")
        alert_container = page.get_by_test_id("stAlertContentError")
        system_status_button = page.get_by_text("System Status")
        vector_store_doc_count = page.get_by_test_id("documents_in_vector_store_count")
        page.goto("http://localhost:8501")
        chat_input.type(inference_string)
        chat_submit_button.click()
        expect(alert_container).to_have_text(
            ("Request failed: 503 Server Error: Service Unavailable for url:")
            + ("http://inference_service:8000/chat/domain-expert/")
        )
        system_status_button.click()
        print(vector_store_doc_count.text_content())
        expect(vector_store_doc_count).to_have_text("0")
        sleep(5)


# Ingest document
# curl -X POST http://127.0.0.1:8003/ingestion/documents/   -H "Content-Type: application/json"   -d '{"documents":
#  ["https://istqb-documents.s3.eu-central-003.backblazeb2.com/ISTQB_CTAL-TM_Syllabus_v3.0.pdf"]}'
#
# Assert System UI shows: Add data-testid
# System Status
# Inference Service Health
# ✓ Connected
# Documents in vector store: 197 - stMarkdownContainer
# Loaded Documents
# ✅ ISTQB_CTAL-TM_Syllabus_v3.0.pdf — Document processing completed
# Inference request: How can I calculate the total cost of quality?
# Assert response xyz
