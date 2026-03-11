from sqlalchemy import Enum, Column, String
from sqlalchemy.orm import declarative_base

from src.shared.constants import DocumentStatus


Base = declarative_base()


class DBDMSDocument(Base):
    __tablename__ = "documents"
    doc_hash = Column(String, primary_key=True)
    doc_name = Column(String, nullable=False)
    status = Column(Enum(DocumentStatus), nullable=False)
