import pytest
from unittest.mock import Mock, patch
from src.console_ui import ConsoleUI
from langchain_community.vectorstores import FAISS
from src.domain_expert import domain_expert, run_chat_loop
from src.chain_manager import ChainManager
from src.exceptions import ExitApp
from src.constants import Error, ChatbotMode, EXIT_WORDS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


class TestDomainExpert:
    @pytest.fixture
    def mock_console_ui(self):
        return Mock(spec=ConsoleUI)

    @pytest.fixture
    def mock_vectordb(self):
        return Mock(spec=FAISS)

    @pytest.fixture
    def mock_chain_manager(self):
        return Mock(spec=ChainManager)

    @pytest.fixture
    def mock_conversational_retrieval_chain(self):
        return Mock(spec=ConversationalRetrievalChain)

    @patch("src.domain_expert.ChainManager")
    @patch("src.domain_expert.run_chat_loop")
    def test_domain_expert_init_success(
        self,
        mock_run_chat_loop,
        mock_chain_manager_class,
        mock_console_ui,
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
        domain_expert(mock_console_ui, mock_vectordb)

        # Assert
        mock_chain_manager_class.assert_called_once_with(mock_vectordb)
        mock_console_ui.show_info_message.assert_any_call("\n🧠 Loading LLM.")
        mock_chain_manager.get_llm.assert_called_once()
        mock_console_ui.show_info_message.assert_any_call("\n⛓ Setting up Chain.")
        mock_chain_manager.get_conversationalRetrievalChain.assert_called_once()
        mock_run_chat_loop.assert_called_once()

    @patch("src.domain_expert.ChainManager")
    def test_domain_expert_chain_manager_init_error(
        self, mock_chain_manager_class, mock_console_ui, mock_vectordb
    ):
        # Arrange
        mock_chain_manager_class.side_effect = ValueError(
            "Error setting up Chain Manager"
        )

        # Act
        with pytest.raises(ExitApp):
            domain_expert(mock_console_ui, mock_vectordb)

        # Assert
        mock_console_ui.show_error.assert_called_once_with(
            Error.EXCEPTION, mock_chain_manager_class.side_effect
        )

    @patch("src.domain_expert.ChainManager")
    def test_domain_expert_chain_manager_get_llm_error(
        self,
        mock_chain_manager_class,
        mock_console_ui,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        mock_chain_manager_class.return_value = mock_chain_manager
        mock_chain_manager.get_llm.side_effect = Exception("Error getting LLM")

        # Act
        with pytest.raises(ExitApp):
            domain_expert(mock_console_ui, mock_vectordb)

        # Assert
        mock_console_ui.show_error.assert_called_once_with(
            Error.EXCEPTION, mock_chain_manager.get_llm.side_effect
        )

    @patch("src.domain_expert.ChainManager")
    def test_domain_expert_chain_manager_get_conversationalRetrievalChain_error(
        self,
        mock_chain_manager_class,
        mock_console_ui,
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
            domain_expert(mock_console_ui, mock_vectordb)

        # Assert
        mock_console_ui.show_error.assert_called_once_with(
            Error.EXCEPTION,
            mock_chain_manager.get_conversationalRetrievalChain.side_effect,
        )

    def test_run_chat_loop_exit_words(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # Arrange
        for exit_word in EXIT_WORDS:
            mock_console_ui.get_user_input.return_value = exit_word

            # Act
            with pytest.raises(ExitApp):
                run_chat_loop(
                    mock_console_ui,
                    mock_chain_manager,
                    mock_conversational_retrieval_chain,
                )

            # Assert
            mock_console_ui.show_welcome_mode.assert_called_with(
                ChatbotMode.DOMAIN_EXPERT
            )

    def test_run_chat_loop_mode_switch(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # Arrange
        mock_console_ui.get_user_input.return_value = "mode"

        # Act
        run_chat_loop(
            mock_console_ui,
            mock_chain_manager,
            mock_conversational_retrieval_chain,
        )

        # Assert
        mock_console_ui.show_mode_switch.assert_called_once()

    def test_run_chat_loop_not_a_question(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # TODO Arrange - How can I make it so the first input is an enter and the second a string
        mock_console_ui.get_user_input.return_value = ""
        mock_console_ui.get_user_input.return_value = "quit"

        # Act
        run_chat_loop(
            mock_console_ui,
            mock_chain_manager,
            mock_conversational_retrieval_chain,
        )

        # Assert
        mock_console_ui.show_error.assert_called_once_with(Error.NOT_A_QUESTION)
