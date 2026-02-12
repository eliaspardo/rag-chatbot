import pytest
from unittest.mock import Mock, patch

from src.core import file_loader as file_loader_module
from src.core.exceptions import ConfigurationException
from src.core.file_loader import FileLoader


class TestFileLoader:
    def test_init_raises_when_pdf_path_empty(self, monkeypatch):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "   ")

        with pytest.raises(ConfigurationException, match="PDF_PATH is empty"):
            FileLoader()

    def test_init_raises_when_s3_and_temp_folder_missing(self, monkeypatch):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "s3://bucket/a.pdf")
        monkeypatch.setattr(file_loader_module, "AWS_TEMP_FOLDER", "")

        with pytest.raises(ConfigurationException, match="AWS_TEMP_FOLDER"):
            FileLoader()

    @patch("src.core.file_loader.shutil.rmtree")
    @patch("src.core.file_loader.os.path.exists", return_value=True)
    def test_init_cleans_temp_folder_for_s3(
        self, mock_exists, mock_rmtree, monkeypatch
    ):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "s3://bucket/a.pdf")
        monkeypatch.setattr(file_loader_module, "AWS_TEMP_FOLDER", "/tmp/aws-temp")

        FileLoader()

        mock_rmtree.assert_called_once_with("/tmp/aws-temp")

    def test_load_pdf_file_raises_for_non_pdf(self, monkeypatch):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "local.pdf")
        loader = FileLoader()

        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_pdf_file("a.txt")

    @patch("src.core.file_loader.os.path.exists", return_value=True)
    def test_load_pdf_file_returns_existing_local_path(self, mock_exists, monkeypatch):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "local.pdf")
        loader = FileLoader()

        assert loader.load_pdf_file("/tmp/local.pdf") == "/tmp/local.pdf"

    @patch("src.core.file_loader.os.path.exists", return_value=False)
    def test_load_pdf_file_raises_for_missing_local_path(
        self, mock_exists, monkeypatch
    ):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "local.pdf")
        loader = FileLoader()

        with pytest.raises(FileNotFoundError, match="File not found"):
            loader.load_pdf_file("/tmp/local.pdf")

    @patch("src.core.file_loader.os.path.exists", return_value=False)
    def test_load_pdf_file_downloads_s3(self, mock_exists, monkeypatch):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "s3://bucket/a.pdf")
        monkeypatch.setattr(file_loader_module, "AWS_TEMP_FOLDER", "/tmp/aws-temp")
        loader = FileLoader()
        loader._download_file_from_s3 = Mock(return_value="/tmp/aws-temp/file.pdf")

        out = loader.load_pdf_file("s3://bucket/a.pdf")

        assert out == "/tmp/aws-temp/file.pdf"
        loader._download_file_from_s3.assert_called_once_with("s3://bucket/a.pdf")

    @patch("src.core.file_loader.os.path.exists", return_value=False)
    def test_load_pdf_file_converts_https_then_downloads(
        self, mock_exists, monkeypatch
    ):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "https://x")
        monkeypatch.setattr(file_loader_module, "AWS_TEMP_FOLDER", "/tmp/aws-temp")
        loader = FileLoader()
        loader._convert_https_to_s3_uri = Mock(return_value="s3://bucket/a.pdf")
        loader._download_file_from_s3 = Mock(return_value="/tmp/aws-temp/file.pdf")

        out = loader.load_pdf_file("https://bucket.s3.us-east-1.amazonaws.com/a.pdf")

        assert out == "/tmp/aws-temp/file.pdf"
        loader._convert_https_to_s3_uri.assert_called_once()
        loader._download_file_from_s3.assert_called_once_with("s3://bucket/a.pdf")

    def test_convert_https_to_s3_virtual_hosted_style(self, monkeypatch):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "local.pdf")
        loader = FileLoader()

        uri = loader._convert_https_to_s3_uri(
            "https://my-bucket.s3.us-east-1.amazonaws.com/folder/a.pdf"
        )

        assert uri == "s3://my-bucket/folder/a.pdf"

    def test_convert_https_to_s3_path_style(self, monkeypatch):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "local.pdf")
        loader = FileLoader()

        uri = loader._convert_https_to_s3_uri(
            "https://s3.us-east-1.amazonaws.com/my-bucket/folder/a.pdf"
        )

        assert uri == "s3://my-bucket/folder/a.pdf"

    def test_convert_https_to_s3_invalid_url(self, monkeypatch):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "local.pdf")
        loader = FileLoader()

        with pytest.raises(ValueError, match="Invalid S3 URL"):
            loader._convert_https_to_s3_uri("https:///")

    @patch("src.core.file_loader.os.makedirs")
    @patch("src.core.file_loader.os.path.exists", return_value=False)
    @patch("src.core.file_loader.boto3.client")
    def test_download_file_from_s3(
        self, mock_boto_client, mock_exists, mock_makedirs, monkeypatch
    ):
        monkeypatch.setattr(file_loader_module, "PDF_PATH", "local.pdf")
        monkeypatch.setattr(file_loader_module, "AWS_TEMP_FOLDER", "/tmp/aws-temp")
        loader = FileLoader()
        s3_client = mock_boto_client.return_value
        loader._generate_random_local_filename = Mock(
            return_value="/tmp/aws-temp/r.pdf"
        )
        loader._extract_S3_bucket_and_key = Mock(return_value=("bucket", "k/a.pdf"))

        out = loader._download_file_from_s3("s3://bucket/k/a.pdf")

        assert out == "/tmp/aws-temp/r.pdf"
        mock_makedirs.assert_called_once_with("/tmp/aws-temp")
        s3_client.download_file.assert_called_once_with(
            "bucket", "k/a.pdf", "/tmp/aws-temp/r.pdf"
        )
