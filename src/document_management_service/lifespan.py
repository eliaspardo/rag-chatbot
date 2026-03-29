"""Lifespan context manager for the Document Management Service FastAPI application."""

from contextlib import asynccontextmanager
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.shared.env_loader import load_environment
from src.document_management_service.models import Base

logger = logging.getLogger(__name__)


load_environment()
DMS_DATABASE_URL = os.getenv("DMS_DATABASE_URL", "sqlite:///:memory:")


@asynccontextmanager
async def lifespan(app):
    """Initialize the database engine and session factory on application startup."""
    engine = create_engine(DMS_DATABASE_URL)
    Base.metadata.create_all(engine)  # Create tables if they don't exist
    app.state.Session = sessionmaker(bind=engine)
    yield
