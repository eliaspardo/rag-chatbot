from typing import List

from sqlalchemy import select
from src.shared.constants import DocumentStatus, SetDocumentResult
from src.shared.exceptions import DocumentHashConflictException
from src.shared.models import DMSDocument
from src.document_management_service.models import DBDMSDocument
from sqlalchemy.orm import Session


class DBClient:
    def __init__(self, session: Session):
        self.session: Session = session

    def get_document_name(self, doc_hash) -> str | None:
        return self.session.execute(
            select(DBDMSDocument.doc_name).where(DBDMSDocument.doc_hash == doc_hash)
        ).scalar()

    def get_document_status(self, doc_hash) -> DocumentStatus | None:
        return DocumentStatus(
            self.session.execute(
                select(DBDMSDocument.status).where(DBDMSDocument.doc_hash == doc_hash)
            ).scalar()
        )

    def get_documents(self) -> List[DMSDocument] | None:
        rows = self.session.execute(select(DBDMSDocument)).scalars().all()
        if not rows:
            return None
        return [DMSDocument.model_validate(row, from_attributes=True) for row in rows]

    def set_document_status(
        self, doc_hash, doc_name, status
    ) -> tuple[DMSDocument, SetDocumentResult]:
        row = self.session.query(DBDMSDocument).filter_by(doc_hash=doc_hash).first()
        if row:
            if row.doc_name != doc_name:
                raise DocumentHashConflictException()
            row.status = status
            self.session.commit()
            return (
                DMSDocument.model_validate(row, from_attributes=True),
                SetDocumentResult.UPDATED,
            )
        db_dms_document = DBDMSDocument(
            doc_hash=doc_hash, doc_name=doc_name, status=status
        )
        self.session.add(db_dms_document)
        self.session.commit()
        dms_document = DMSDocument.model_validate(db_dms_document, from_attributes=True)
        return dms_document, SetDocumentResult.CREATED
