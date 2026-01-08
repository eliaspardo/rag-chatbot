import pytest
from unittest.mock import Mock, patch
from src.ui.console_ui import ConsoleUI
from langchain_community.vectorstores import FAISS
from src.core.exam_prep import exam_prep, run_chat_loop
from src.core.chain_manager import ChainManager
from src.core.exceptions import ExitApp
from src.core.constants import Error, ChatbotMode, EXIT_WORDS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from src.core.prompts import exam_prep_question_prompt, exam_prep_answer_prompt


class TestExamPrep:
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

    @patch("src.core.exam_prep.ChainManager")
    @patch("src.core.exam_prep.run_chat_loop")
    def test_exam_prep_init_success(
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
        exam_prep(mock_console_ui, mock_vectordb)

        # Assert
        mock_chain_manager_class.assert_called_once_with(mock_vectordb)
        mock_console_ui.show_info_message.assert_any_call("\n🧠 Loading LLM.")
        mock_chain_manager.get_llm.assert_called_once()
        mock_console_ui.show_info_message.assert_any_call("\n⛓ Setting up Chains.")
        mock_chain_manager.get_conversationalRetrievalChain.assert_any_call(
            mock_llm, {"prompt": exam_prep_question_prompt}
        )
        mock_chain_manager.get_conversationalRetrievalChain.assert_any_call(
            mock_llm, {"prompt": exam_prep_answer_prompt}
        )
        mock_run_chat_loop.assert_called_once()

    @patch("src.core.exam_prep.ChainManager")
    def test_exam_prep_chain_manager_init_error(
        self, mock_chain_manager_class, mock_console_ui, mock_vectordb
    ):
        # Arrange
        mock_chain_manager_class.side_effect = ValueError(
            "Error setting up Chain Manager"
        )

        # Act
        with pytest.raises(ExitApp):
            exam_prep(mock_console_ui, mock_vectordb)

        # Assert
        mock_console_ui.show_error.assert_called_once_with(
            Error.EXCEPTION, mock_chain_manager_class.side_effect
        )

    @patch("src.core.exam_prep.ChainManager")
    def test_exam_prep_chain_manager_get_llm_error(
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
            exam_prep(mock_console_ui, mock_vectordb)

        # Assert
        mock_console_ui.show_error.assert_called_once_with(
            Error.EXCEPTION, mock_chain_manager.get_llm.side_effect
        )

    @patch("src.core.exam_prep.ChainManager")
    def test_exam_prep_chain_manager_get_conversationalRetrievalChain_error_on_first_chain(
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
            exam_prep(mock_console_ui, mock_vectordb)

        # Assert
        mock_console_ui.show_error.assert_called_once_with(
            Error.EXCEPTION,
            mock_chain_manager.get_conversationalRetrievalChain.side_effect,
        )

    @patch("src.core.exam_prep.ChainManager")
    def test_exam_prep_chain_manager_get_conversationalRetrievalChain_error_on_second_chain(
        self,
        mock_chain_manager_class,
        mock_console_ui,
        mock_vectordb,
        mock_chain_manager,
    ):
        # Arrange
        exception = Exception("Error getting ConversationalRetrievalChain")
        mock_chain_manager_class.return_value = mock_chain_manager
        mock_chain_manager.get_llm.return_value = Mock()
        mock_chain_manager.get_conversationalRetrievalChain.side_effect = [
            Mock(),
            exception,
        ]

        # Act
        with pytest.raises(ExitApp):
            exam_prep(mock_console_ui, mock_vectordb)

        # Assert
        mock_console_ui.show_error.assert_called_once_with(
            Error.EXCEPTION,
            exception,
        )

    def test_run_chat_loop_exit_words_on_first_input_loop(
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
                    mock_conversational_retrieval_chain,
                )

            # Assert
            mock_console_ui.show_welcome_mode.assert_called_with(ChatbotMode.EXAM_PREP)

    def test_run_chat_loop_mode_switch_on_first_input_loop(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # Arrange
        mock_console_ui.get_user_input.return_value = "mode"

        # Act
        run_chat_loop(
            mock_console_ui,
            mock_chain_manager,
            mock_conversational_retrieval_chain,
            mock_conversational_retrieval_chain,
        )

        # Assert
        mock_console_ui.show_mode_switch.assert_called_once()

    def test_run_chat_loop_not_a_topic_on_first_input_loop(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # Arrange
        mock_console_ui.get_user_input.side_effect = [""]

        # Act
        with pytest.raises(StopIteration):
            run_chat_loop(
                mock_console_ui,
                mock_chain_manager,
                mock_conversational_retrieval_chain,
                mock_conversational_retrieval_chain,
            )

        # Assert
        mock_console_ui.show_error.assert_called_once_with(Error.NOT_A_TOPIC)

    def test_run_chat_loop_question_error(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # Arrange
        mock_console_ui.get_user_input.side_effect = ["Sample Topic", "quit"]
        mock_chain_manager.ask_question.side_effect = Exception(
            "Error getting question"
        )

        # Act
        with pytest.raises(ExitApp):
            run_chat_loop(
                mock_console_ui,
                mock_chain_manager,
                mock_conversational_retrieval_chain,
                mock_conversational_retrieval_chain,
            )

        # Assert
        mock_console_ui.show_error.assert_called_with(
            Error.EXCEPTION, exception=mock_chain_manager.ask_question.side_effect
        )

    def test_run_chat_loop_exit_words_on_second_input_loop(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # Arrange
        for exit_word in EXIT_WORDS:
            mock_console_ui.get_user_input.side_effect = ["Sample Topic", exit_word]

            # Act
            with pytest.raises(ExitApp):
                run_chat_loop(
                    mock_console_ui,
                    mock_chain_manager,
                    mock_conversational_retrieval_chain,
                    mock_conversational_retrieval_chain,
                )

            # Assert
            mock_console_ui.show_welcome_mode.assert_called_with(ChatbotMode.EXAM_PREP)

    def test_run_chat_loop_mode_switch_on_second_input_loop(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # Arrange
        mock_console_ui.get_user_input.side_effect = ["Sample Topic", "mode"]

        # Act
        run_chat_loop(
            mock_console_ui,
            mock_chain_manager,
            mock_conversational_retrieval_chain,
            mock_conversational_retrieval_chain,
        )

        # Assert
        mock_console_ui.show_mode_switch.assert_called_once()

    def test_run_chat_loop_no_user_answer_on_second_input_loop(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # Arrange
        mock_console_ui.get_user_input.side_effect = ["Sample Topic", ""]
        mock_chain_manager.ask_question.return_value = "Sample Question"

        # Act
        with pytest.raises(StopIteration):
            run_chat_loop(
                mock_console_ui,
                mock_chain_manager,
                mock_conversational_retrieval_chain,
                mock_conversational_retrieval_chain,
            )

        # Assert
        mock_console_ui.show_error.assert_called_once_with(Error.NO_USER_ANSWER)

    def test_run_chat_loop_answer_error(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # Arrange
        exception = Exception("Error getting answer")
        mock_console_ui.get_user_input.side_effect = ["Sample Topic", "Sample Answer"]
        mock_chain_manager.ask_question.side_effect = ["Sample Question", exception]

        # Act
        with pytest.raises(Exception):
            run_chat_loop(
                mock_console_ui,
                mock_chain_manager,
                mock_conversational_retrieval_chain,
                mock_conversational_retrieval_chain,
            )

        # Assert
        mock_console_ui.show_error.assert_called_with(
            Error.EXCEPTION, exception=exception
        )

    def test_run_chat_loop_keyboard_exception(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # Arrange
        exception = KeyboardInterrupt()
        mock_console_ui.get_user_input.side_effect = ["Sample Topic", exception]
        mock_chain_manager.ask_question.side_effect = ["Sample Question"]

        # Act
        with pytest.raises(ExitApp):
            run_chat_loop(
                mock_console_ui,
                mock_chain_manager,
                mock_conversational_retrieval_chain,
                mock_conversational_retrieval_chain,
            )

    def test_run_chat_loop_success(
        self, mock_console_ui, mock_chain_manager, mock_conversational_retrieval_chain
    ):
        # Arrange
        mock_console_ui.get_user_input.side_effect = ["Sample Topic", "Sample Answer"]
        mock_chain_manager.ask_question.side_effect = [
            "Sample Question",
            "Sample Answer",
        ]

        # Act
        with pytest.raises(StopIteration):
            run_chat_loop(
                mock_console_ui,
                mock_chain_manager,
                mock_conversational_retrieval_chain,
                mock_conversational_retrieval_chain,
            )

        # Assert
        mock_console_ui.show_answer.assert_called_with("Sample Answer")
