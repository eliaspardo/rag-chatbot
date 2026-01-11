from typing import Dict, Union
import uuid

from langchain_core.vectorstores import VectorStore

from src.core.domain_expert_core import DomainExpertCore
from src.core.exam_prep_core import ExamPrepCore

import logging

logger = logging.getLogger(__name__)


class Session:
    def __init__(self):
        self.id = str(uuid.uuid4())


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
        self.sessions[session.id] = session
        return session

    def create_exam_prep_session(self):
        session = ExamPrepSession(self.vectordb)
        self.sessions[session.id] = session
        return session

    def remove_session(self, session: Session):
        del self.sessions[session.id]

    def get_session_by_id(self, id: str):
        return self.sessions.get(id)

    def remove_session_by_id(self, id: str):
        del self.sessions[id]

    def get_domain_expert_session(self, id: str = None) -> DomainExpertSession:
        if not id:
            # If no session id, create session.
            logger.info("No session id provided. Creating new Domain Expert session.")
            return self.create_domain_expert_session()
        session = self.sessions.get(id)
        if not session:
            # The client might have a stale id - generate a new session
            # TODO - show some kind of message to the user saying chat history is gone?
            logger.warning("Session id not found. Creating new Domain Expert session.")
            return self.create_domain_expert_session()
        if not isinstance(session, DomainExpertSession):
            # If incorrect type, create new session
            logger.warning("Session mismatch. Creating new Domain Expert session.")
            # TODO - show some kind of message saying session is of another type
            return self.create_domain_expert_session()
        return session

    def get_exam_prep_session(self, id: str = None) -> ExamPrepSession:
        if not id:
            # If no session id, create session.
            logger.info("No session id provided. Creating new Exam Prep session.")
            return self.create_exam_prep_session()
        session = self.sessions.get(id)
        if not session:
            # The client might have a stale id - generate a new session
            logger.warning("Session id not found. Creating new Exam Prep session.")
            # TODO - show some kind of message saying chat history is gone
            return self.create_exam_prep_session()
        if not isinstance(session, ExamPrepSession):
            # If incorrect type, create new session
            logger.warning("Session mismatch. Creating new Exam Prep session.")
            # TODO - show some kind of message saying session is of another type
            return self.create_exam_prep_session()
        return session
