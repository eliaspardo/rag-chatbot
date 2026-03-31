import pytest
import chromadb
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.document_management_service.models import DBDMSDocument

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
        finally:
            session.close()
            engine.dispose()

    def test_e2e_flow(self, clear_vector_db, clear_dms_db):
        pass


# Run tests - delete/clear vector store and dms db? How to deal with this in production?
#
# No document ingested - 503
#
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
