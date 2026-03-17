from typing import Dict, Tuple, Optional
import uuid

from langchain_core.vectorstores import VectorStore

from src.inference_service.core.domain_expert_core import DomainExpertCore

import logging

logger = logging.getLogger(__name__)


class DomainExpertSession:
    def __init__(self, vectordb: VectorStore):
        self.session_id = str(uuid.uuid4())
        self.domain_expert_core = DomainExpertCore(vectordb)


class SessionManager:
    def __init__(self, vectordb: VectorStore):
        self.sessions: Dict[str, DomainExpertSession] = {}
        self.vectordb = vectordb

    def get_sessions(self) -> Dict[str, DomainExpertSession]:
        return self.sessions

    def create_domain_expert_session(self):
        session = DomainExpertSession(self.vectordb)
        self.sessions[session.session_id] = session
        return session

    def remove_session(self, session: DomainExpertSession):
        self.sessions.pop(session.session_id, None)

    def get_session_by_id(self, session_id: str):
        return self.sessions.get(session_id)

    def remove_session_by_id(self, session_id: str):
        self.sessions.pop(session_id, None)

    def get_domain_expert_session(
        self, session_id: str = None
    ) -> Tuple[DomainExpertSession, Optional[str]]:
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
