import pytest
from unittest.mock import Mock, patch
from src.core.domain_expert_core import DomainExpertCore
from src.ui.console_ui import ConsoleUI
from src.ui.domain_expert_ui import run_domain_expert_chat_loop
from src.core.exceptions import ExitApp
from src.core.constants import Error, ChatbotMode, EXIT_WORDS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


class TestDomainExpertUi:
    @pytest.fixture
    def mock_console_ui(self):
        return Mock(spec=ConsoleUI)

    @pytest.fixture
    def mock_domain_expert_core(self):
        return Mock(spec=DomainExpertCore)

    def test_run_chat_loop_exit_words(self, mock_console_ui, mock_domain_expert_core):
        # Arrange
        for exit_word in EXIT_WORDS:
            mock_console_ui.get_user_input.return_value = exit_word

            # Act
            with pytest.raises(ExitApp):
                run_domain_expert_chat_loop(mock_console_ui, mock_domain_expert_core)

            # Assert
            mock_console_ui.show_welcome_mode.assert_called_with(
                ChatbotMode.DOMAIN_EXPERT
            )

    def test_run_chat_loop_mode_switch(self, mock_console_ui, mock_domain_expert_core):
        # Arrange
        mock_console_ui.get_user_input.return_value = "mode"

        # Act
        run_domain_expert_chat_loop(mock_console_ui, mock_domain_expert_core)

        # Assert
        mock_console_ui.show_mode_switch.assert_called_once()

    def test_run_chat_loop_not_a_question(
        self, mock_console_ui, mock_domain_expert_core
    ):
        # Arrange
        mock_console_ui.get_user_input.side_effect = ["", "quit"]

        # Act
        with pytest.raises(ExitApp):
            run_domain_expert_chat_loop(mock_console_ui, mock_domain_expert_core)

        # Assert
        mock_console_ui.show_error.assert_called_once_with(Error.NOT_A_QUESTION)

    def test_run_chat_loop_question_error(
        self, mock_console_ui, mock_domain_expert_core
    ):
        # Arrange
        mock_console_ui.get_user_input.side_effect = ["Sample Question", "quit"]
        mock_domain_expert_core.ask_question.side_effect = Exception(
            "Error getting answer"
        )

        # Act
        with pytest.raises(ExitApp):
            run_domain_expert_chat_loop(mock_console_ui, mock_domain_expert_core)

        # Assert
        mock_console_ui.show_error.assert_called_with(
            Error.EXCEPTION, exception=mock_domain_expert_core.ask_question.side_effect
        )

    def test_run_chat_loop_keyboard_exception(
        self, mock_console_ui, mock_domain_expert_core
    ):
        # Arrange
        exception = KeyboardInterrupt()
        mock_console_ui.get_user_input.side_effect = ["Sample Question", exception]
        mock_domain_expert_core.ask_question.side_effect = ["Sample Answer"]

        # Act
        with pytest.raises(ExitApp):
            run_domain_expert_chat_loop(mock_console_ui, mock_domain_expert_core)

    def test_run_chat_loop_success(self, mock_console_ui, mock_domain_expert_core):
        # Arrange
        mock_console_ui.get_user_input.side_effect = ["Sample Question", "quit"]
        mock_domain_expert_core.ask_question.return_value = "Sample Answer"

        # Act
        with pytest.raises(ExitApp):
            run_domain_expert_chat_loop(mock_console_ui, mock_domain_expert_core)

        # Assert
        mock_console_ui.show_answer.assert_called_once_with(
            mock_domain_expert_core.ask_question.return_value
        )
