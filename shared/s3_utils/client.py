"""
boto3 S3 client — singleton with connection pooling.
Thread-safe: boto3 clients are safe to share across threads.
"""

import logging
from functools import lru_cache

import boto3
from botocore.config import Config as BotoConfig

from .config import get_s3_config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_s3_client():
    """
    Returns a singleton boto3 S3 client configured from environment.
    Uses connection pooling (max 25 connections) and retry logic.

    Returns:
        boto3.client: Configured S3 client, or None if S3 is disabled/misconfigured.
    """
    config = get_s3_config()

    if not config.is_s3_enabled:
        logger.info("S3 storage disabled (STORAGE_BACKEND=%s)", config.storage_backend)
        return None

    if not config.has_credentials:
        logger.warning("S3 credentials missing — falling back to local storage")
        return None

    if not config.bucket_name:
        logger.warning("S3_BUCKET_NAME not set — falling back to local storage")
        return None

    boto_config = BotoConfig(
        max_pool_connections=25,
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=10,
        read_timeout=30,
    )

    client_kwargs = {
        "service_name": "s3",
        "aws_access_key_id": config.access_key_id,
        "aws_secret_access_key": config.secret_access_key,
        "config": boto_config,
    }

    # Region is optional — only set if provided
    if config.region:
        client_kwargs["region_name"] = config.region

    # Custom endpoint for MinIO/localstack testing
    if config.endpoint_url:
        client_kwargs["endpoint_url"] = config.endpoint_url

    try:
        client = boto3.client(**client_kwargs)
        # Quick validation: check bucket exists
        client.head_bucket(Bucket=config.bucket_name)
        logger.info(
            "S3 client initialized — bucket=%s, prefix=%s",
            config.bucket_name,
            config.agent_prefix,
        )
        return client
    except Exception as e:
        logger.error("Failed to initialize S3 client: %s", e)
        return None


def get_s3_resource():
    """
    Returns a boto3 S3 resource (higher-level API) for multipart uploads.
    Used for large file uploads (FAISS indexes > 100MB).
    """
    config = get_s3_config()

    if not config.is_s3_enabled or not config.has_credentials:
        return None

    resource_kwargs = {
        "service_name": "s3",
        "aws_access_key_id": config.access_key_id,
        "aws_secret_access_key": config.secret_access_key,
    }

    if config.region:
        resource_kwargs["region_name"] = config.region

    if config.endpoint_url:
        resource_kwargs["endpoint_url"] = config.endpoint_url

    try:
        return boto3.resource(**resource_kwargs)
    except Exception as e:
        logger.error("Failed to create S3 resource: %s", e)
        return None
