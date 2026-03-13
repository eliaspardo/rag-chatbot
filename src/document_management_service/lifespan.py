from contextlib import asynccontextmanager
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.shared.env_loader import load_environment

logger = logging.getLogger(__name__)


load_environment()
DMS_DATABASE_URL = os.getenv("DMS_DATABASE_URL", "sqlite:///:memory:")


@asynccontextmanager
async def lifespan(app):
    engine = create_engine(DMS_DATABASE_URL)
    app.state.Session = sessionmaker(bind=engine)
    yield
