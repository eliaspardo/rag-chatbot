"""Session manager for tracking domain expert conversation sessions."""

from typing import Dict, Tuple, Optional
import uuid

from langchain_core.vectorstores import VectorStore

from src.inference_service.core.domain_expert_core import DomainExpertCore

import logging

logger = logging.getLogger(__name__)


class DomainExpertSession:
    """Represents a single user conversation session with a DomainExpertCore instance."""

    def __init__(self, vectordb: VectorStore):
        self.session_id = str(uuid.uuid4())
        self.domain_expert_core = DomainExpertCore(vectordb)


class SessionManager:
    """Manages the lifecycle of DomainExpertSession instances keyed by session ID."""

    def __init__(self, vectordb: VectorStore):
        self.sessions: Dict[str, DomainExpertSession] = {}
        self.vectordb = vectordb

    def get_sessions(self) -> Dict[str, DomainExpertSession]:
        """Return the current mapping of session IDs to DomainExpertSession objects."""
        return self.sessions

    def create_domain_expert_session(self):
        """Create and register a new DomainExpertSession, then return it."""
        session = DomainExpertSession(self.vectordb)
        self.sessions[session.session_id] = session
        return session

    def remove_session(self, session: DomainExpertSession):
        """Remove a session from the registry by its session object."""
        self.sessions.pop(session.session_id, None)

    def get_session_by_id(self, session_id: str):
        """Look up and return a session by ID, or None if not found."""
        return self.sessions.get(session_id)

    def remove_session_by_id(self, session_id: str):
        """Remove a session from the registry by its session ID."""
        self.sessions.pop(session_id, None)

    def get_domain_expert_session(
        self, session_id: str = None
    ) -> Tuple[DomainExpertSession, Optional[str]]:
        """Return an existing session or create a new one; include a warning message if the session was missing."""
        if not session_id:
            # If no session id, create session.
            logger.info("No session id provided. Creating new Domain Expert session.")
            return self.create_domain_expert_session(), None
        session = self.sessions.get(session_id)
        if not session:
            # The client might have a stale id - generate a new session
            system_message = "Session id not found. Creating new Domain Expert session. Chat history will be lost"
            logger.warning(system_message)
            return self.create_domain_expert_session(), system_message
        return session, None
