import os
import logging
import shutil
from src.env_loader import load_environment
import boto3
import uuid

load_environment()
AWS_TEMP_FOLDER = os.getenv("AWS_TEMP_FOLDER")
AWS_REGION = os.getenv("AWS_REGION")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
# AWS_ACCESS_KEY_ID
# AWS_SECRET_ACCESS_KEY

logger = logging.getLogger(__name__)


class FileLoader:
    def __init__(self):
        # Cleanup temp directory if needed
        if not os.path.exists(AWS_TEMP_FOLDER):
            os.mkdir(AWS_TEMP_FOLDER)
        else:
            shutil.rmtree(AWS_TEMP_FOLDER)
            os.mkdir(AWS_TEMP_FOLDER)

    def load_pdf_file(self, file_path: str) -> str:
        if not file_path.endswith(".pdf"):
            raise ValueError(f"Unsupported file type: {file_path}")
        if file_path.startswith(
            "s3://"
        ):  # If file is an S3 URL, download the file and store locally
            try:
                # Download the file from S3 and store locally
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
            s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
            # Check if file exists in s3 bucket
            response = s3_client.head_object(Bucket=AWS_BUCKET_NAME, Key=file_path)
            if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                raise Exception(f"File not found in S3 bucket: {file_path}")
            # Download the file to the temporary directory
            if not os.path.exists(AWS_TEMP_FOLDER):
                os.makedirs(AWS_TEMP_FOLDER)
            # Randomly generate a file name
            temp_file_name = f"{uuid.uuid4()}-{os.path.basename(file_path)}"
            temp_file_path = os.path.join(AWS_TEMP_FOLDER, temp_file_name)
            s3_client.download_file(
                AWS_BUCKET_NAME,
                file_path,
                temp_file_path,
            )
            return temp_file_path
        except Exception as e:
            logger.error(f"Error downloading file from S3: {e}")
            raise Exception(f"Error downloading file from S3: {e}")
