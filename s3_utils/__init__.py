"""
Shared S3 utility module for VCS AI Agents.
Used by all agents under PROD_SETUP for S3 read/write operations.

Usage:
    from s3_utils.config import get_s3_config
    from s3_utils.client import get_s3_client
    from s3_utils.operations import upload_file, download_file, upload_bytes, download_bytes, list_objects, delete_object
    from s3_utils.helpers import agent_prefix, project_path, dated_log_path
"""

__version__ = "1.0.0"
