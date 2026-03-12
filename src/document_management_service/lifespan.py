from contextlib import asynccontextmanager
import logging
import os
from sqlalchemy.orm import sessionmaker

from src.shared.env_loader import load_environment

logger = logging.getLogger(__name__)


load_environment()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///:memory:")


@asynccontextmanager
async def lifespan(app):
    app.state.Session = sessionmaker(bind=DATABASE_URL)
    yield
