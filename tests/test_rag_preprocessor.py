import pytest
from unittest.mock import patch
from src.rag_preprocessor import RAGPreprocessor


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
        texts = rag_preprocessor.load_pdf_text("tests/data/pdf-test.pdf")
        full_text = "".join(texts)
        assert "Congratulations" in full_text
