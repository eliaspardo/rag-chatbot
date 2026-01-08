import pytest
from unittest.mock import Mock, patch
from langchain_community.vectorstores import FAISS
from src.core.domain_expert_core import DomainExpertCore
from src.core.chain_manager import ChainManager
from src.core.exceptions import DomainExpertSetupException
from src.core.prompts import condense_question_prompt, domain_expert_prompt


class TestDomainExpertCore:
    @pytest.fixture
    def mock_vectordb(self):
        return Mock(spec=FAISS)

    @pytest.fixture
    def mock_chain_manager(self):
        return Mock(spec=ChainManager)

    def _build_core(self, mock_chain_manager_class, mock_chain_manager, mock_vectordb):
        mock_chain_manager_class.return_value = mock_chain_manager
        mock_llm = Mock()
        mock_chain_manager.get_llm.return_value = mock_llm
        mock_qa_chain = Mock()
        mock_chain_manager.get_conversationalRetrievalChain.return_value = mock_qa_chain
        return DomainExpertCore(mock_vectordb)

    @patch("src.core.domain_expert_core.ChainManager")
    def test_domain_expert_init_success(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Act
        core = self._build_core(
            mock_chain_manager_class, mock_chain_manager, mock_vectordb
        )

        # Assert
        mock_chain_manager_class.assert_called_once_with(mock_vectordb)
        mock_chain_manager.get_llm.assert_called_once()
        mock_chain_manager.get_conversationalRetrievalChain.assert_called_once_with(
            mock_chain_manager.get_llm.return_value,
            {"prompt": domain_expert_prompt},
            condense_question_prompt=condense_question_prompt,
        )
        assert core.chain_manager == mock_chain_manager
        assert (
            core.qa_chain
            == mock_chain_manager.get_conversationalRetrievalChain.return_value
        )

    @patch("src.core.domain_expert_core.ChainManager")
    def test_domain_expert_init_chain_manager_error(
        self,
        mock_chain_manager_class,
        mock_vectordb,
    ):
        # Arrange
        mock_chain_manager_class.side_effect = ValueError(
            "Error instantiating Chain Manager"
        )

        # Act
        with pytest.raises(DomainExpertSetupException):
            DomainExpertCore(mock_vectordb)

    @patch("src.core.domain_expert_core.ChainManager")
    def test_domain_expert_get_llm_error(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        mock_chain_manager_class.return_value = mock_chain_manager
        mock_chain_manager.get_llm.side_effect = Exception("Error getting LLM")

        # Act
        with pytest.raises(DomainExpertSetupException):
            DomainExpertCore(mock_vectordb)

    @patch("src.core.domain_expert_core.ChainManager")
    def test_domain_expert_get_conversational_chain_error(
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
        with pytest.raises(DomainExpertSetupException):
            DomainExpertCore(mock_vectordb)

    @patch("src.core.domain_expert_core.ChainManager")
    def test_ask_question_success(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        core = self._build_core(
            mock_chain_manager_class, mock_chain_manager, mock_vectordb
        )
        mock_chain_manager.ask_question.return_value = "This is the answer"

        # Act
        answer = core.ask_question("This is the question")

        # Assert
        assert answer == "This is the answer"
        mock_chain_manager.ask_question.assert_called_once_with(
            "This is the question", core.qa_chain
        )

    @patch("src.core.domain_expert_core.ChainManager")
    def test_ask_question_failure(
        self,
        mock_chain_manager_class,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        core = self._build_core(
            mock_chain_manager_class, mock_chain_manager, mock_vectordb
        )
        mock_chain_manager.ask_question.side_effect = Exception(
            "Exception getting answer"
        )

        # Act
        with pytest.raises(DomainExpertSetupException):
            core.ask_question("This is the question")
