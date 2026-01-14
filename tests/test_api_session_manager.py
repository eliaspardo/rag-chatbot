import pytest
from unittest.mock import Mock, patch

from src.api.session_manager import SessionManager, DomainExpertSession


class TestSessionManager:
    @pytest.fixture
    def mock_vectordb(self):
        return Mock()

    @patch("src.api.session_manager.DomainExpertCore")
    def test_get_domain_expert_session_creates_when_missing(
        self, mock_domain_expert_core, mock_vectordb
    ):
        manager = SessionManager(mock_vectordb)

        session, system_message = manager.get_domain_expert_session()

        assert isinstance(session, DomainExpertSession)
        assert system_message is None
        assert session.session_id in manager.sessions
        mock_domain_expert_core.assert_called_once_with(mock_vectordb)

    @patch("src.api.session_manager.DomainExpertCore")
    def test_get_domain_expert_session_stale_id_creates_new(
        self, mock_domain_expert_core, mock_vectordb
    ):
        manager = SessionManager(mock_vectordb)

        session, system_message = manager.get_domain_expert_session("missing")

        assert isinstance(session, DomainExpertSession)
        assert system_message == (
            "Session id not found. Creating new Domain Expert session. Chat history will be lost"
        )
        assert session.session_id in manager.sessions
        mock_domain_expert_core.assert_called_once_with(mock_vectordb)

    @patch("src.api.session_manager.DomainExpertCore")
    def test_remove_and_get_session_by_id(self, mock_domain_expert_core, mock_vectordb):
        manager = SessionManager(mock_vectordb)
        session = manager.create_domain_expert_session()

        assert manager.get_session_by_id(session.session_id) == session

        manager.remove_session_by_id(session.session_id)

        assert manager.get_session_by_id(session.session_id) is None
