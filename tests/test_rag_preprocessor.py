import pytest
from unittest.mock import patch
from src.rag_preprocessor import RAGPreprocessor
from langchain.schema import Document


TEST_PDF = "tests/data/pdf-test.pdf"
STRING_LIST = ["test", "string", "for", "testing"]
EMPTY_LIST = []
STRING_LIST_EMPTY_CHUNKS = ["test", "    ", "for", "   testing   "]


class TestRagPreprocessor:
    @pytest.fixture
    def rag_preprocessor(self):
        return RAGPreprocessor()

    def test_load_pdf_text_no_file(self, rag_preprocessor):
        with pytest.raises(Exception, match="Error reading PDF file"):
            rag_preprocessor.load_pdf_text("no_file")

    @patch("src.rag_preprocessor.fitz.open")
    def test_load_pdf_text_default_no_file(self, mock_fitz_open, rag_preprocessor):
        mock_fitz_open.side_effect = FileNotFoundError("No such file or directory")
        with pytest.raises(Exception, match="Error reading PDF file"):
            rag_preprocessor.load_pdf_text()

    def test_load_pdf_text_successful(self, rag_preprocessor):
        texts = rag_preprocessor.load_pdf_text(TEST_PDF)
        full_text = "".join(texts)
        assert "Congratulations" in full_text

    def test_split_text_to_docs_success(self, rag_preprocessor):
        documents = rag_preprocessor.split_text_to_docs(STRING_LIST)
        assert len(documents) > 0
        for document in documents:
            print("document")
            assert isinstance(document, Document)

    def test_split_text_to_docs_empty_list(self, rag_preprocessor):
        documents = rag_preprocessor.split_text_to_docs(EMPTY_LIST)
        assert len(documents) == 0

    def test_split_text_to_docs_empty_chunks(self, rag_preprocessor):
        documents = rag_preprocessor.split_text_to_docs(STRING_LIST_EMPTY_CHUNKS)
        assert all(document.page_content.strip() for document in documents)
