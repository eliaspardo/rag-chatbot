import pytest
from unittest.mock import Mock, patch
from src.core.exceptions import (
    ExitApp,
    FaissException,
    NoDocumentsException,
    VectorStoreException,
)
from src.core.constants import Error, EXIT_WORDS, ChatbotMode
from src.ui.console_ui import ConsoleUI
from src.main import main, run_app, run_chat_loop


class TestMain:
    @pytest.fixture
    def mock_console_ui_instance(self):
        return Mock(spec=ConsoleUI)

    @pytest.fixture
    def mock_vectordb_instance(self):
        return Mock()

    @patch("src.main.run_app")
    @patch("src.main.ConsoleUI")
    def test_main_shows_welcome(self, mock_console_ui, mock_run_app):
        # Arrange
        mock_run_app.side_effect = None
        # Act
        main()

        # Assert
        mock_console_ui.return_value.show_welcome.assert_called_once()

    @patch("sys.exit")
    @patch("src.main.run_app")
    @patch("src.main.ConsoleUI")
    def test_main_exitApp_error(self, mock_console_ui, mock_run_app, mock_sys_exit):
        # Arrange
        mock_run_app.side_effect = ExitApp()
        mock_sys_exit.side_effect = None
        # Act
        main()

        # Assert
        mock_console_ui.return_value.show_exit_message.assert_called_once()

    @patch("src.main.run_chat_loop")
    @patch("src.main.prepare_vector_store")
    @patch("src.main.FileLoader")
    @patch("src.main.get_rag_preprocessor")
    def test_run_app_success(
        self,
        mock_get_rag_preprocessor,
        mock_file_loader,
        mock_prepare_vector_store,
        mock_run_chat_loop,
        mock_console_ui_instance,
    ):
        # Arrange
        rag_preprocessor = mock_get_rag_preprocessor.return_value
        file_loader = mock_file_loader.return_value
        vectordb = Mock()
        mock_prepare_vector_store.return_value = vectordb

        # Act
        run_app(mock_console_ui_instance)

        # Assert
        mock_prepare_vector_store.assert_called_once_with(
            rag_preprocessor=rag_preprocessor,
            file_loader=file_loader,
            progress_callback=mock_console_ui_instance.show_info_message,
        )
        mock_run_chat_loop.assert_called_once_with(mock_console_ui_instance, vectordb)

    @patch("src.main.FileLoader")
    def test_run_app_file_loader_error(
        self, mock_file_loader, mock_console_ui_instance
    ):
        # Arrange
        exception = Exception("bad config")
        mock_file_loader.side_effect = exception

        # Act
        with pytest.raises(ExitApp):
            run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_error.assert_called_once_with(
            Error.EXCEPTION, exception
        )

    @patch("src.main.prepare_vector_store")
    @patch("src.main.FileLoader")
    @patch("src.main.get_rag_preprocessor")
    def test_run_app_no_docs_error(
        self,
        mock_get_rag_preprocessor,
        mock_file_loader,
        mock_prepare_vector_store,
        mock_console_ui_instance,
    ):
        # Arrange
        mock_prepare_vector_store.side_effect = NoDocumentsException()

        # Act
        with pytest.raises(ExitApp):
            run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_error.assert_called_once_with(Error.NO_DOCUMENTS)

    @pytest.mark.parametrize("exception", [FaissException(), VectorStoreException()])
    @patch("src.main.prepare_vector_store")
    @patch("src.main.FileLoader")
    @patch("src.main.get_rag_preprocessor")
    def test_run_app_vector_store_errors(
        self,
        mock_get_rag_preprocessor,
        mock_file_loader,
        mock_prepare_vector_store,
        mock_console_ui_instance,
        exception,
    ):
        # Arrange
        mock_prepare_vector_store.side_effect = exception

        # Act
        with pytest.raises(ExitApp):
            run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_error.assert_called_once_with(
            Error.EXCEPTION, exception
        )

    @patch("src.main.prepare_vector_store")
    @patch("src.main.FileLoader")
    @patch("src.main.get_rag_preprocessor")
    def test_run_app_generic_error(
        self,
        mock_get_rag_preprocessor,
        mock_file_loader,
        mock_prepare_vector_store,
        mock_console_ui_instance,
    ):
        # Arrange
        exception = Exception("unexpected")
        mock_prepare_vector_store.side_effect = exception

        # Act
        with pytest.raises(ExitApp):
            run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_error.assert_called_once_with(
            Error.EXCEPTION, exception
        )

    def test_run_chat_loop_exit_words(
        self, mock_console_ui_instance, mock_vectordb_instance
    ):
        # Arrange
        for exit_word in EXIT_WORDS:
            mock_console_ui_instance.get_operational_mode_selection.return_value = (
                exit_word
            )

            # Act
            with pytest.raises(ExitApp):
                run_chat_loop(mock_console_ui_instance, mock_vectordb_instance)

            # Assert
            mock_console_ui_instance.show_operational_mode_selection.assert_called()

    @patch("src.main.DomainExpertCore")
    @patch("src.main.run_domain_expert_chat_loop")
    @patch("src.main.run_exam_prep_chat_loop")
    @patch("src.main.ExamPrepCore")
    def test_run_chat_loop_select_modes(
        self,
        mock_exam_prep_core,
        mock_run_exam_prep_chat_loop,
        mock_run_domain_expert_chat_loop,
        mock_domain_expert_core,
        mock_console_ui_instance,
        mock_vectordb_instance,
    ):
        # Arrange
        modes = [ChatbotMode.DOMAIN_EXPERT, ChatbotMode.EXAM_PREP]
        mock_exam_prep_core.return_value = Mock()
        mock_run_domain_expert_chat_loop.return_value = None
        mock_run_exam_prep_chat_loop.return_value = None
        mock_domain_expert_core.return_value = Mock()
        mock_console_ui_instance.get_operational_mode_selection.side_effect = [
            "1",
            "2",
            "quit",
        ]

        # Act
        with pytest.raises(ExitApp):
            run_chat_loop(mock_console_ui_instance, mock_vectordb_instance)

        # Assert
        for mode in modes:
            mock_console_ui_instance.show_entering_mode.assert_any_call(mode)
        mock_domain_expert_core.assert_called_once_with(mock_vectordb_instance)
        mock_run_domain_expert_chat_loop.assert_called_once_with(
            mock_console_ui_instance, mock_domain_expert_core.return_value
        )
        mock_exam_prep_core.assert_called_once_with(mock_vectordb_instance)
        mock_run_exam_prep_chat_loop.assert_called_once_with(
            mock_console_ui_instance, mock_exam_prep_core.return_value
        )

    def test_run_chat_loop_invalid_input(
        self,
        mock_console_ui_instance,
        mock_vectordb_instance,
    ):
        # Arrange
        mock_console_ui_instance.get_operational_mode_selection.side_effect = [
            "Invalid input"
        ]
        with pytest.raises(StopIteration):
            # Act
            run_chat_loop(mock_console_ui_instance, mock_vectordb_instance)

            # Assert
            mock_console_ui_instance.show_error.assert_called_with(Error.INVALID_MODE)
