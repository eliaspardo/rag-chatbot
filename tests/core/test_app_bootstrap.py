from unittest.mock import Mock, call, patch

import pytest
from langchain.schema import Document

from src.core import app_bootstrap as app_bootstrap_module
from src.core.app_bootstrap import prepare_vector_store
from src.core.exceptions import ConfigurationException, NoDocumentsException


class TestAppBootstrap:
    @patch("src.core.app_bootstrap.os.path.exists", return_value=True)
    def test_prepare_vector_store_uses_existing_db(self, mock_exists):
        rag_preprocessor = Mock()
        rag_preprocessor.load_vector_store.return_value = "vectordb"
        file_loader = Mock()
        progress = Mock()

        out = prepare_vector_store(
            rag_preprocessor=rag_preprocessor,
            file_loader=file_loader,
            progress_callback=progress,
            db_dir="db",
        )

        assert out == "vectordb"
        file_loader.load_pdf_file.assert_not_called()
        rag_preprocessor.create_vector_store.assert_not_called()
        rag_preprocessor.load_vector_store.assert_called_once_with(db_dir="db")
        progress.assert_has_calls(
            [call("📦 Using existing vector store."), call("📶 Loading vector store.")]
        )

    @patch("src.core.app_bootstrap.os.path.exists", return_value=False)
    def test_prepare_vector_store_aggregates_docs_from_multiple_pdfs(self, mock_exists):
        rag_preprocessor = Mock()
        file_loader = Mock()
        progress = Mock()
        doc_a = Document(page_content="a")
        doc_b = Document(page_content="b")
        rag_preprocessor.load_pdf_text.side_effect = [["raw-a"], ["raw-b"]]
        rag_preprocessor.split_text_to_docs.side_effect = [[doc_a], [doc_b]]
        rag_preprocessor.load_vector_store.return_value = "vectordb"
        file_loader.load_pdf_file.side_effect = ["/tmp/a.pdf", "/tmp/b.pdf"]

        with patch.object(app_bootstrap_module, "PDF_PATH", "a.pdf, b.pdf"):
            out = prepare_vector_store(
                rag_preprocessor=rag_preprocessor,
                file_loader=file_loader,
                progress_callback=progress,
                db_dir="db",
            )

        assert out == "vectordb"
        file_loader.load_pdf_file.assert_has_calls([call("a.pdf"), call("b.pdf")])
        rag_preprocessor.create_vector_store.assert_called_once_with(
            [doc_a, doc_b], db_dir="db"
        )
        rag_preprocessor.load_vector_store.assert_called_once_with(db_dir="db")
        progress.assert_any_call("🏭 Creating vector store.")
        progress.assert_any_call("✅ Vector DB created and saved.")

    @patch("src.core.app_bootstrap.os.path.exists", return_value=False)
    def test_prepare_vector_store_empty_pdf_path(self, mock_exists):
        rag_preprocessor = Mock()
        file_loader = Mock()

        with patch.object(app_bootstrap_module, "PDF_PATH", "   "):
            with pytest.raises(ConfigurationException, match="PDF_PATH is empty"):
                prepare_vector_store(
                    rag_preprocessor=rag_preprocessor,
                    file_loader=file_loader,
                    db_dir="db",
                )

    @patch("src.core.app_bootstrap.os.path.exists", return_value=False)
    def test_prepare_vector_store_raises_when_no_documents_after_split(
        self, mock_exists
    ):
        rag_preprocessor = Mock()
        file_loader = Mock()
        file_loader.load_pdf_file.return_value = "/tmp/a.pdf"
        rag_preprocessor.load_pdf_text.return_value = ["raw"]
        rag_preprocessor.split_text_to_docs.return_value = []

        with patch.object(app_bootstrap_module, "PDF_PATH", "a.pdf"):
            with pytest.raises(NoDocumentsException, match="No documents found"):
                prepare_vector_store(
                    rag_preprocessor=rag_preprocessor,
                    file_loader=file_loader,
                    db_dir="db",
                )
