from contextlib import asynccontextmanager
import logging

from src.document_management_service.db_client import DBClient

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # Startup, get DB client
    app.state.db_client = DBClient()

    yield

    # Shutdown
    print("Cleaning up...")
