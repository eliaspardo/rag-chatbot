import pytest
from unittest.mock import Mock, patch
from langchain_community.vectorstores import FAISS
from src.core.domain_expert_core import setup_domain_expert_chain
from src.core.chain_manager import ChainManager
from src.core.exceptions import ExitApp
from src.core.constants import Error
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


class TestDomainExpertCore:
    @pytest.fixture
    def mock_vectordb(self):
        return Mock(spec=FAISS)

    @pytest.fixture
    def mock_chain_manager(self):
        return Mock(spec=ChainManager)

    @pytest.fixture
    def mock_conversational_retrieval_chain(self):
        return Mock(spec=ConversationalRetrievalChain)

    @patch("src.core.domain_expert_core.ChainManager")
    @patch("src.core.domain_expert.run_chat_loop")
    def test_domain_expert_init_success(
        self,
        mock_run_chat_loop,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        mock_chain_manager_class.return_value = mock_chain_manager
        mock_llm = Mock()
        mock_chain_manager.get_llm.return_value = mock_llm
        mock_qa_chain = Mock()
        mock_chain_manager.get_conversationalRetrievalChain.return_value = mock_qa_chain
        mock_run_chat_loop.return_value = None

        # Act
        setup_domain_expert_chain(mock_chain_manager)

        # Assert
        mock_chain_manager_class.assert_called_once_with(mock_vectordb)
        mock_chain_manager.get_llm.assert_called_once()
        mock_chain_manager.get_conversationalRetrievalChain.assert_called_once()
        mock_run_chat_loop.assert_called_once()

    

    @patch("src.core.domain_expert_core.ChainManager")
    def test_domain_expert_chain_manager_get_llm_error(
        self,
        mock_chain_manager_class,        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        mock_chain_manager_class.return_value = mock_chain_manager
        mock_chain_manager.get_llm.side_effect = Exception("Error getting LLM")

        # Act
        with pytest.raises(ExitApp):
            setup_domain_expert_chain(mock_chain_manager)

        # TODO Assert

    @patch("src.core.domain_expert_core.ChainManager")
    def test_domain_expert_chain_manager_get_conversationalRetrievalChain_error(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        mock_chain_manager_class.return_value = mock_chain_manager
        mock_chain_manager.get_llm.return_value = Mock()
        mock_chain_manager.get_conversationalRetrievalChain.side_effect = Exception(
            "Error getting ConversationalRetrievalChain"
        )

        # Act
        with pytest.raises(ExitApp):
            setup_domain_expert_chain(mock_chain_manager)

        # TODO Assert
