import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.document_management_service.db_client import DBClient
from src.document_management_service.models import Base, DBDMSDocument
from src.shared.constants import DocumentStatus, SetDocumentResult
from src.shared.exceptions import DocumentHashConflictException
from src.shared.models import DMSDocument


sample_hash = "d41d8cd98f00b204e9800998ecf8427e"
sample_doc_name = "Test doc name"
sample_status = DocumentStatus.PENDING


class TestDBClient:
    @pytest.fixture(scope="function")
    def db_client(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()
        yield DBClient(session)

    def test_get_document_name_success(self, db_client):
        # Seed db with DMSDocument
        db_client.session.add(
            DBDMSDocument(
                doc_hash=sample_hash, doc_name=sample_doc_name, status=sample_status
            )
        )
        result = db_client.get_document_name(sample_hash)
        assert result == sample_doc_name

    def test_get_document_name_not_found(self, db_client):
        # No document in the db
        result = db_client.get_document_name(sample_hash)
        assert result is None

    def test_get_document_status_success(self, db_client):
        # Seed db with DMSDocument
        db_client.session.add(
            DBDMSDocument(
                doc_hash=sample_hash, doc_name=sample_doc_name, status=sample_status
            )
        )
        result = db_client.get_document_status(sample_hash)
        assert result == sample_status

    def test_get_document_status_not_found(self, db_client):
        # No document in the db
        result = db_client.get_document_status(sample_hash)
        assert result is None

    def test_get_documents_success(self, db_client):
        # Seed db with DMSDocuments
        document_1 = DMSDocument(
            doc_hash=sample_hash, doc_name=sample_doc_name, status=sample_status
        )
        document_2 = DMSDocument(
            doc_hash="1234567890",
            doc_name="document_2",
            status=DocumentStatus.COMPLETED,
        )
        db_client.session.add(DBDMSDocument(**document_1.model_dump()))
        db_client.session.add(DBDMSDocument(**document_2.model_dump()))
        result = db_client.get_documents()
        assert result == [document_1, document_2]

    def test_get_documents_no_documents(self, db_client):
        # No document in the db
        result = db_client.get_documents()
        assert result is None

    def test_set_document_status_not_existing_success(self, db_client):
        # No document in the db
        document, result = db_client.set_document_status(
            sample_hash, sample_doc_name, sample_status
        )
        assert document == DMSDocument(
            doc_hash=sample_hash, doc_name=sample_doc_name, status=sample_status
        )
        assert result == SetDocumentResult.CREATED

    def test_set_document_status_existing_success(self, db_client):
        # Seed db with DMSDocument
        db_client.session.add(
            DBDMSDocument(
                doc_hash=sample_hash,
                doc_name=sample_doc_name,
                status=DocumentStatus.PENDING,
            )
        )
        document, result = db_client.set_document_status(
            sample_hash, sample_doc_name, DocumentStatus.COMPLETED
        )
        assert document == DMSDocument(
            doc_hash=sample_hash,
            doc_name=sample_doc_name,
            status=DocumentStatus.COMPLETED,
        )
        assert result == SetDocumentResult.UPDATED

    def test_set_document_status_existing_mismatch(self, db_client):
        # Seed db with DMSDocument
        db_client.session.add(
            DBDMSDocument(
                doc_hash=sample_hash,
                doc_name="test_name_1",
                status=DocumentStatus.PENDING,
            )
        )
        with pytest.raises(DocumentHashConflictException):
            document, result = db_client.set_document_status(
                sample_hash, "test_name_2", DocumentStatus.COMPLETED
            )
