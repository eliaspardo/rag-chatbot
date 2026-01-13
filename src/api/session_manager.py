from typing import Dict, Tuple, Union, Optional
import uuid

from langchain_core.vectorstores import VectorStore

from src.core.domain_expert_core import DomainExpertCore
from src.core.exam_prep_core import ExamPrepCore

import logging

logger = logging.getLogger(__name__)


class Session:
    def __init__(self):
        self.session_id = str(uuid.uuid4())


class DomainExpertSession(Session):
    def __init__(self, vectordb: VectorStore):
        super().__init__()
        self.domain_expert_core = DomainExpertCore(vectordb)


class ExamPrepSession(Session):
    def __init__(self, vectordb: VectorStore):
        super().__init__()
        self.exam_prep_core = ExamPrepCore(vectordb)


class SessionManager:
    def __init__(self, vectordb: VectorStore):
        self.sessions: Dict[str, Union[DomainExpertSession, ExamPrepSession]] = {}
        self.vectordb = vectordb

    def get_sessions(self) -> Dict[str, Union[DomainExpertSession, ExamPrepSession]]:
        return self.sessions

    def create_domain_expert_session(self):
        session = DomainExpertSession(self.vectordb)
        self.sessions[session.session_id] = session
        return session

    def create_exam_prep_session(self):
        session = ExamPrepSession(self.vectordb)
        self.sessions[session.session_id] = session
        return session

    def remove_session(self, session: Session):
        del self.sessions[session.session_id]

    def get_session_by_id(self, session_id: str):
        return self.sessions.get(session_id)

    def remove_session_by_id(self, session_id: str):
        del self.sessions[session_id]

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
        if not isinstance(session, DomainExpertSession):
            # If incorrect type, create new session
            system_message = "Session mismatch. Creating new Domain Expert session. Chat history will be lost"
            logger.warning(system_message)
            return self.create_domain_expert_session(), system_message
        return session, None

    def get_exam_prep_session(
        self, session_id: str = None
    ) -> Tuple[ExamPrepSession, Optional[str]]:
        if not session_id:
            # If no session id, create session.
            logger.info("No session id provided. Creating new Exam Prep session.")
            return self.create_exam_prep_session(), None
        session = self.sessions.get(session_id)
        if not session:
            # The client might have a stale id - generate a new session
            system_message = "Session id not found. Creating new Exam Prep session."
            logger.warning(system_message)
            return self.create_exam_prep_session(), system_message
        if not isinstance(session, ExamPrepSession):
            # If incorrect type, create new session
            system_message = "Session mismatch. Creating new Exam Prep session."
            logger.warning(system_message)
            return self.create_exam_prep_session(), system_message
        return session, None
