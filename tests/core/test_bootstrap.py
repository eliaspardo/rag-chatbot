from unittest.mock import Mock, call, patch

import pytest
from langchain.schema import Document

from src.ingestion_service import bootstrap as bootstrap_module
from src.ingestion_service.bootstrap import prepare_vector_store
from src.shared.exceptions import ConfigurationException, NoDocumentsException


class TestBootstrap:
    def test_prepare_vector_store_uses_existing_db(self):
        vector_store_builder = Mock()
        vector_store_builder.collection_has_documents.return_value = True
        file_loader = Mock()
        progress = Mock()

        prepare_vector_store(
            vector_store_builder=vector_store_builder,
            file_loader=file_loader,
            progress_callback=progress,
        )

        file_loader.load_pdf_file.assert_not_called()
        vector_store_builder.create_vector_store.assert_not_called()
        progress.assert_called_once_with("📦 Using existing vector store.")

    def test_prepare_vector_store_aggregates_docs_from_multiple_pdfs(self):
        vector_store_builder = Mock()
        vector_store_builder.collection_has_documents.return_value = False
        file_loader = Mock()
        progress = Mock()
        doc_a = Document(page_content="a")
        doc_b = Document(page_content="b")
        vector_store_builder.load_pdf_text.side_effect = [["raw-a"], ["raw-b"]]
        vector_store_builder.split_text_to_docs.side_effect = [[doc_a], [doc_b]]
        file_loader.load_pdf_file.side_effect = ["/tmp/a.pdf", "/tmp/b.pdf"]

        with patch.object(bootstrap_module, "PDF_PATH", "a.pdf, b.pdf"):
            prepare_vector_store(
                vector_store_builder=vector_store_builder,
                file_loader=file_loader,
                progress_callback=progress,
            )

        file_loader.load_pdf_file.assert_has_calls([call("a.pdf"), call("b.pdf")])
        vector_store_builder.create_vector_store.assert_called_once_with([doc_a, doc_b])
        progress.assert_any_call("🏭 Creating vector store.")
        progress.assert_any_call("✅ Vector DB created and saved.")

    def test_prepare_vector_store_empty_pdf_path(self):
        vector_store_builder = Mock()
        vector_store_builder.collection_has_documents.return_value = False
        file_loader = Mock()

        with patch.object(bootstrap_module, "PDF_PATH", "   "):
            with pytest.raises(ConfigurationException, match="PDF_PATH is empty"):
                prepare_vector_store(
                    vector_store_builder=vector_store_builder, file_loader=file_loader
                )

    def test_prepare_vector_store_raises_when_no_documents_after_split(self):
        vector_store_builder = Mock()
        vector_store_builder.collection_has_documents.return_value = False
        file_loader = Mock()
        file_loader.load_pdf_file.return_value = "/tmp/a.pdf"
        vector_store_builder.load_pdf_text.return_value = ["raw"]
        vector_store_builder.split_text_to_docs.return_value = []

        with patch.object(bootstrap_module, "PDF_PATH", "a.pdf"):
            with pytest.raises(NoDocumentsException, match="No documents found"):
                prepare_vector_store(
                    vector_store_builder=vector_store_builder, file_loader=file_loader
                )
