import pytest
from unittest.mock import Mock, patch
from src.exceptions import ExitApp
from src.constants import Error
from src.console_ui import ConsoleUI
from src.main import main, run_app, run_chat_loop
from src.rag_preprocessor import RAGPreprocessor


class TestMain:

    @pytest.fixture
    def mock_console_ui_instance(self):
        return Mock(spec=ConsoleUI)

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
    @patch("src.main.RAGPreprocessor")
    @patch("os.path.exists", return_value=True)
    def test_run_app_path_exists_uses_existing_vector_store(
        self,
        os_path_exists,
        mock_rag_preprocessor,
        mock_run_chat_loop,
        mock_console_ui_instance,
    ):
        # Arrange
        mock_rag_preprocessor.load_vector_store.return_value = None
        mock_run_chat_loop.return_value = None

        # Act
        run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_info_message.assert_any_call(
            "📦 Using existing vector store."
        )
        mock_console_ui_instance.show_info_message.assert_any_call(
            ("📶 Loading vector store.")
        )

    @patch("src.main.RAGPreprocessor")
    @patch("os.path.exists", return_value=True)
    def test_run_app_path_exists_error_loading_store(
        self,
        os_path_exists,
        mock_rag_preprocessor,
        mock_console_ui_instance,
    ):
        # Arrange
        mock_rag_preprocessor.return_value.load_vector_store.side_effect = Exception(
            "Error loading vector store"
        )

        # Act
        with pytest.raises(ExitApp):
            run_app(mock_console_ui_instance)

        # Assert
        mock_console_ui_instance.show_error.assert_called_with(
            Error.EXCEPTION,
            mock_rag_preprocessor.return_value.load_vector_store.side_effect,
        )
