import os
import logging
import shutil
from typing import Tuple
from src.env_loader import load_environment
import boto3
import uuid
from urllib.parse import urlparse


load_environment()
PDF_PATH = os.getenv("PDF_PATH")
AWS_TEMP_FOLDER = os.getenv("AWS_TEMP_FOLDER", "")
AWS_REGION = os.getenv("AWS_REGION")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

logger = logging.getLogger(__name__)


class FileLoader:
    def __init__(self):
        pdf_paths = [p.strip() for p in (PDF_PATH or "").split(",") if p.strip()]
        self.using_s3 = any(path.startswith("s3://") for path in pdf_paths)
        self.using_s3 = any(path.startswith("https://") for path in pdf_paths)
        self.using_s3 = any(path.startswith("http://") for path in pdf_paths)

        if self.using_s3:
            if not AWS_TEMP_FOLDER:
                raise ValueError(
                    "AWS_TEMP_FOLDER is not set but S3 PDF paths are configured."
                )
            # Ensure a clean temporary directory for S3 downloads
            if os.path.exists(AWS_TEMP_FOLDER):
                shutil.rmtree(AWS_TEMP_FOLDER)
        elif AWS_TEMP_FOLDER and os.path.exists(AWS_TEMP_FOLDER):
            # No S3 usage configured; remove any leftover temp directory
            shutil.rmtree(AWS_TEMP_FOLDER)

    def load_pdf_file(self, file_path: str) -> str:
        if not file_path.endswith(".pdf"):
            raise ValueError(f"Unsupported file type: {file_path}")
        if file_path.startswith("https://") or file_path.startswith("http://"):
            file_path = self._convert_https_to_s3_uri(file_path)
        if file_path.startswith(
            "s3://"
        ):  # If file is an S3 URL, download the file and store locally
            try:
                file_path = self._download_file_from_s3(file_path)
                return file_path
            except Exception as e:
                logger.error(f"Error downloading file from S3: {e}")
                raise
        # If file exists return the file path
        if os.path.exists(file_path):
            return file_path
        raise FileNotFoundError(f"File not found: {file_path}")

    def _download_file_from_s3(self, file_path: str) -> str:
        try:
            s3_client = boto3.client(
                "s3",
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                endpoint_url=AWS_ENDPOINT_URL,
            )
            # Download the file to the temporary directory
            if not os.path.exists(AWS_TEMP_FOLDER):
                os.makedirs(AWS_TEMP_FOLDER)

            temp_file_path = self._generate_random_local_filename(file_path)
            bucket, key = self._extract_S3_bucket_and_key(file_path)
            s3_client.download_file(
                bucket,
                key,
                temp_file_path,
            )
            return temp_file_path
        except Exception as e:
            raise Exception(f"Error downloading file from S3: {e}")

    def _extract_S3_bucket_and_key(self, file_path: str) -> Tuple[str, str]:
        parsed = urlparse(file_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return bucket, key

    def _generate_random_local_filename(self, file_path: str) -> str:
        temp_file_name = f"{uuid.uuid4()}-{os.path.basename(file_path)}"
        temp_file_path = os.path.join(AWS_TEMP_FOLDER, temp_file_name)
        return temp_file_path

    def _convert_https_to_s3_uri(self, https_url: str) -> str:
        """Convert S3 HTTPS URL to s3:// URI format."""
        from urllib.parse import urlparse

        parsed = urlparse(https_url)
        hostname = parsed.hostname
        path = parsed.path.lstrip("/")

        if not hostname or not path:
            raise ValueError(f"Invalid S3 URL: {https_url}")

        # Virtual-hosted-style: bucket.s3.region.provider.com
        if ".s3." in hostname or ".s3-" in hostname:
            bucket = hostname.split(".")[0]
            key = path
        # Path-style: s3.region.provider.com/bucket/key
        else:
            parts = path.split("/", 1)
            if not parts[0]:
                raise ValueError(f"Could not extract bucket from URL: {https_url}")
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

        return f"s3://{bucket}/{key}"
