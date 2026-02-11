import os
import logging
import shutil
from src.env_loader import load_environment
import boto3
import uuid

load_environment()
PDF_PATH = os.getenv("PDF_PATH")
AWS_TEMP_FOLDER = os.getenv("AWS_TEMP_FOLDER", "")
AWS_REGION = os.getenv("AWS_REGION")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

logger = logging.getLogger(__name__)


class FileLoader:
    def __init__(self):
        pdf_paths = [p.strip() for p in (PDF_PATH or "").split(",") if p.strip()]
        self.using_s3 = any(path.startswith("s3://") for path in pdf_paths)

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
            # Randomly generate a file name
            temp_file_name = f"{uuid.uuid4()}-{os.path.basename(file_path)}"
            temp_file_path = os.path.join(AWS_TEMP_FOLDER, temp_file_name)
            s3_client.download_file(
                AWS_BUCKET_NAME,
                os.path.basename(file_path),
                temp_file_path,
            )
            return temp_file_path
        except Exception as e:
            raise Exception(f"Error downloading file from S3: {e}")
