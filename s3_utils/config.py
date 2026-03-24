"""
S3 configuration — reads from environment variables.
Each agent loads its own .env, so these values come from whichever agent imports this module.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class S3Config:
    """Immutable S3 configuration loaded from environment."""

    bucket_name: str
    region: str
    access_key_id: str
    secret_access_key: str
    agent_prefix: str  # e.g. "construction-intelligence-agent"
    storage_backend: str  # "s3" or "local"
    endpoint_url: str = ""  # optional, for MinIO/localstack testing

    @property
    def is_s3_enabled(self) -> bool:
        """Returns True if storage backend is set to S3."""
        return self.storage_backend.lower() == "s3"

    @property
    def has_credentials(self) -> bool:
        """Returns True if AWS credentials are configured."""
        return bool(self.access_key_id and self.secret_access_key)


@lru_cache(maxsize=1)
def get_s3_config() -> S3Config:
    """
    Load S3 configuration from environment variables.
    Cached — call once per process lifetime.

    Required .env variables:
        STORAGE_BACKEND=s3          # "s3" or "local"
        S3_BUCKET_NAME=...
        AWS_ACCESS_KEY_ID=...
        AWS_SECRET_ACCESS_KEY=...
        S3_AGENT_PREFIX=...         # e.g. "construction-intelligence-agent"

    Optional:
        S3_REGION=us-east-1
        S3_ENDPOINT_URL=            # for MinIO/localstack
    """
    return S3Config(
        bucket_name=os.getenv("S3_BUCKET_NAME", ""),
        region=os.getenv("S3_REGION", "us-east-1"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        agent_prefix=os.getenv("S3_AGENT_PREFIX", ""),
        storage_backend=os.getenv("STORAGE_BACKEND", "local"),
        endpoint_url=os.getenv("S3_ENDPOINT_URL", ""),
    )
