# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Tuple
import logging as _py_logging

import boto3
import botocore
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_delay, wait_exponential

from nemo.utils import logging
from nemo.utils.s3_dirpath_utils import build_s3_url, is_s3_url

# Suppress boto3 and botocore noisy logs (e.g. "Found credentials in ...")
_py_logging.getLogger('boto3').setLevel(_py_logging.WARNING)
_py_logging.getLogger('botocore').setLevel(_py_logging.WARNING)

try:
    import awscrt
    import s3transfer.crt

    crt_available = True
except ImportError as e:
    crt_available = False

MB = 1024**2
GB = 1024**3

SHARED_MEM_DIR = '/dev/shm'
DEFAULT_CHUNK_SIZE_MB = 64
DEFAULT_MAX_READ_CONCURRENCY = 15
DEFAULT_MAX_WRITE_CONCURRENCY = 10
DEFAULT_CLIENT_CACHE_TTL_SEC = 30 * 60

_ENDPOINT_URL_ENV_KEYS = ('AWS_ENDPOINT_URL', 'TOS_ENDPOINT', 'TOS_ENDPOINT_URL')
_REGION_ENV_KEYS = ('AWS_DEFAULT_REGION', 'TOS_REGION')
_ADDRESSING_STYLE_ENV_KEYS = ('AWS_S3_ADDRESSING_STYLE', 'TOS_ADDRESSING_STYLE')


class S3Utils:
    """
    Utility class for interacting with S3. Handles downloading and uploading to S3, and parsing/formatting S3 urls.
    """

    '''
    Cache boto3 resources/clients per-process and per-thread with TTL to avoid re-creating
    sessions on every sample fetch. This dramatically reduces small-object GET latency while
    keeping token refresh safe via periodic rebuilds.
    '''

    _RESOURCE_CACHE_LOCAL = threading.local()

    @staticmethod
    def s3_path_exists(s3_path: str, match_directory: bool = False) -> bool:
        """
        :s3_path: the path
        :match_directory: if the content is known to be a directory then set it to `True`. Since s3 isn't a file system, paths are funky and the concept of folders doesn't really exist.
        """
        bucket_name, prefix = S3Utils.parse_s3_url(s3_path)
        if not prefix:
            return False

        s3 = S3Utils._get_s3_resource()
        # bucket = s3.Bucket(bucket_name)
        s3_client = s3.meta.client

        try:
            objs = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1, Prefix=prefix).get('Contents', [])
        except s3_client.exceptions.NoSuchBucket:
            return False

        if prefix == '':  # bucket only
            return True

        return len(objs) > 0 and (match_directory or objs[0]['Key'].startswith(prefix))

    @staticmethod
    def remove_object(s3_path: str) -> None:
        s3_client = S3Utils._get_s3_resource(get_client=True)
        bucket, key = S3Utils.parse_s3_url(s3_path)
        s3_client.delete_object(Bucket=bucket, Key=key)

    @staticmethod
    def download_s3_file_to_stream(
        s3_path: str, chunk_size_MB: int = DEFAULT_CHUNK_SIZE_MB, max_concurrency: int = DEFAULT_MAX_READ_CONCURRENCY
    ) -> BytesIO:
        bytes_buffer = BytesIO()

        s3_client = S3Utils._get_s3_resource(get_client=True)
        bucket, key = S3Utils.parse_s3_url(s3_path)
        chunk_size = chunk_size_MB * MB
        config = TransferConfig(multipart_chunksize=chunk_size, max_concurrency=max_concurrency)

        start_time = time.perf_counter()
        _download_fileobj_with_retry(s3_client, bucket, key, bytes_buffer, config)
        elapsed = time.perf_counter() - start_time
        if os.environ.get("NEMO_S3_DOWNLOAD_LOG_LEVEL", "").lower() == "info":
            logging.info(
                f'Time elapsed downloading {s3_path} to file stream with chunk_size={chunk_size_MB}MB '
                f'and max_concurrency={max_concurrency}: {elapsed:.2f} seconds'
            )
        else:
            logging.debug(
                f'Downloaded {s3_path} to file stream in {elapsed:.2f}s '
                f'(chunk_size={chunk_size_MB}MB, max_concurrency={max_concurrency})'
            )

        bytes_buffer.seek(0)
        return bytes_buffer

    @staticmethod
    def download_s3_file_to_path(
        s3_path: str,
        file_path: str,
        chunk_size_MB: int = DEFAULT_CHUNK_SIZE_MB,
        max_concurrency: int = DEFAULT_MAX_READ_CONCURRENCY,
    ) -> None:
        s3_client = S3Utils._get_s3_resource(get_client=True)
        bucket, key = S3Utils.parse_s3_url(s3_path)
        chunk_size = chunk_size_MB * MB
        config = TransferConfig(multipart_chunksize=chunk_size, max_concurrency=max_concurrency)

        logging.info(
            f'Downloading {s3_path} to {file_path} with chunk_size={chunk_size_MB}MB and max_threads={max_concurrency}'
        )
        start_time = time.perf_counter()
        _download_file_with_retry(s3_client, bucket, key, file_path, config)
        logging.info(
            f'Time elapsed downloading {s3_path} to {file_path} with chunk_size={chunk_size_MB}MB '
            f'and max_concurrency={max_concurrency}: {(time.perf_counter() - start_time):.2f} seconds'
        )

    @staticmethod
    def upload_file_stream_to_s3(
        bytes_buffer: BytesIO,
        s3_path: str,
        chunk_size_MB: int = DEFAULT_CHUNK_SIZE_MB,
        max_concurrency: int = DEFAULT_MAX_WRITE_CONCURRENCY,
    ) -> None:
        s3_client = S3Utils._get_s3_resource(get_client=True)
        bucket, key = S3Utils.parse_s3_url(s3_path)
        chunk_size = chunk_size_MB * MB
        config = TransferConfig(multipart_chunksize=chunk_size, max_concurrency=max_concurrency)
        bytes_buffer.seek(0)

        start_time = time.perf_counter()
        _upload_fileobj_with_retry(s3_client, bytes_buffer, bucket, key, config)
        logging.info(
            f'Time elapsed uploading bytes buffer to {s3_path} with chunk_size={chunk_size_MB}MB '
            f'and max_concurrency={max_concurrency}: {(time.perf_counter() - start_time):.2f} seconds'
        )

    @staticmethod
    def upload_file(
        file_path: str,
        s3_path: str,
        chunk_size_MB=DEFAULT_CHUNK_SIZE_MB,
        max_concurrency=DEFAULT_MAX_WRITE_CONCURRENCY,
        remove_file=False,
    ):
        total_size = os.path.getsize(file_path)
        assert total_size > 0, f"file size is zero, {file_path}"

        s3_client = S3Utils._get_s3_resource(get_client=True)
        bucket, key = S3Utils.parse_s3_url(s3_path)

        chunk_size = chunk_size_MB * MB
        config = TransferConfig(
            multipart_threshold=chunk_size, multipart_chunksize=chunk_size, max_concurrency=max_concurrency
        )

        start_time = time.perf_counter()
        _upload_file_with_retry(s3_client, file_path, bucket, key, config)
        if remove_file and os.path.exists(file_path):
            os.remove(file_path)
        logging.info(
            f'Time elapsed uploading file {file_path} of size {(total_size/GB):.1f}GB to {s3_path} with chunk_size={chunk_size_MB}MB '
            f'and max_concurrency={max_concurrency}: {(time.perf_counter() - start_time):.2f} seconds'
        )

    @staticmethod
    def find_files_with_suffix(
        base_path: str,
        suffix: str = None,
        return_key_only: bool = True,
        profile: Optional[str] = None,
        creds: botocore.credentials.Credentials = None,
    ) -> List[str]:
        """
        Returns a list of keys that have the specified suffix
        :param base_path: the root of search
        :param suffix: the suffix to match, case sensitive
        :return: list of keys matching the suffix, relative to the base_path
        """
        s3 = S3Utils._get_s3_resource(profile, creds)
        bucket_name, prefix = S3Utils.parse_s3_url(base_path)

        start_time = time.perf_counter()
        bucket = s3.Bucket(bucket_name)
        objects_list = _scan_objects_with_retry(s3_bucket=bucket, s3_prefix=prefix)
        logging.info(
            f'Time elapsed reading all objects under path {base_path}: {(time.perf_counter() - start_time):.2f} seconds'
        )

        if suffix:
            objects_list = list(filter(lambda o: o.key.endswith(suffix), objects_list))

        if return_key_only:
            return [o.key for o in objects_list]
        else:
            return [S3Utils.build_s3_url(o.bucket_name, o.key) for o in objects_list]

    @staticmethod
    def _get_client_cache_ttl_sec() -> int:
        """
        Parse per-thread S3 client/resource cache TTL from env.

        Semantics for NEMO_S3_CLIENT_CACHE_TTL_SEC:
          - ttl > 0: cache entries expire after ttl seconds.
          - ttl == 0: disable cache usage (always rebuild client/resource).
          - ttl < 0: never expire cached entries.
        """
        raw = os.environ.get("NEMO_S3_CLIENT_CACHE_TTL_SEC", str(DEFAULT_CLIENT_CACHE_TTL_SEC))
        try:
            ttl = int(raw)
        except (TypeError, ValueError):
            ttl = DEFAULT_CLIENT_CACHE_TTL_SEC
        return ttl

    @staticmethod
    def _resolve_client_env_overrides() -> Tuple[Optional[str], Optional[str], Optional[str]]:
        endpoint_url = None
        for key in _ENDPOINT_URL_ENV_KEYS:
            value = os.environ.get(key)
            if value:
                endpoint_url = value
                break

        region_name = None
        for key in _REGION_ENV_KEYS:
            value = os.environ.get(key)
            if value:
                region_name = value
                break

        valid_addressing_styles = {'virtual', 'path'}
        addressing_style = None
        for key in _ADDRESSING_STYLE_ENV_KEYS:
            value = (os.environ.get(key) or '').strip().lower()
            if value in valid_addressing_styles:
                addressing_style = value
                break

        return endpoint_url, region_name, addressing_style

    @staticmethod
    def _make_cache_key(
        *,
        profile: str,
        creds: botocore.credentials.Credentials,
        get_client: bool,
        endpoint_url: Optional[str],
        region_name: Optional[str],
        config: Dict[str, Any],
    ) -> Hashable:
        if isinstance(creds, dict):
            creds_fp = (
                creds.get("AccessKeyId"),
                creds.get("SecretAccessKey"),
                creds.get("SessionToken"),
            )
        else:
            creds_fp = repr(creds) if creds is not None else None

        return (
            profile,
            creds_fp,
            get_client,
            endpoint_url,
            region_name,
            repr(config),
        )

    @staticmethod
    def _build_resource(
        *,
        profile: str,
        creds: botocore.credentials.Credentials,
        session,
        config_obj: botocore.config.Config,
        endpoint_url: Optional[str],
        region_name: Optional[str],
    ):
        if profile is not None and creds is not None:
            raise ValueError('Please provide profile or creds or neither, not both.')

        if profile is not None:
            boto_session = boto3.Session(profile_name=profile, region_name=region_name)
            return boto_session.resource('s3', config=config_obj, endpoint_url=endpoint_url)

        if creds is not None:
            return boto3.Session(region_name=region_name).resource(
                's3',
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                config=config_obj,
                endpoint_url=endpoint_url,
            )

        boto_session = session if session is not None else boto3.Session(region_name=region_name)
        return boto_session.resource('s3', config=config_obj, endpoint_url=endpoint_url)

    @staticmethod
    def _get_s3_resource(
        profile: str = None,
        creds: botocore.credentials.Credentials = None,
        get_client: bool = False,
        session=None,
        config=None,
        refresh: bool = False,
    ):
        cfg_kwargs: Dict[str, Any] = dict(config or {})
        cfg_kwargs.setdefault('max_pool_connections', 30)

        endpoint_cfg = cfg_kwargs.pop('endpoint_url', None)
        region_cfg = cfg_kwargs.pop('region_name', None)
        s3_cfg = dict(cfg_kwargs.pop('s3', {}) or {})

        endpoint_env, region_env, addressing_style_env = S3Utils._resolve_client_env_overrides()
        endpoint_url = endpoint_cfg or endpoint_env
        region_name = region_cfg or region_env

        if endpoint_url and 'signature_version' not in cfg_kwargs:
            cfg_kwargs['signature_version'] = 's3v4'

        if addressing_style_env and 'addressing_style' not in s3_cfg:
            s3_cfg['addressing_style'] = addressing_style_env
        elif endpoint_url and 'addressing_style' not in s3_cfg:
            # Most TOS endpoints work reliably with virtual-hosted style.
            s3_cfg['addressing_style'] = 'virtual'

        if s3_cfg:
            cfg_kwargs['s3'] = s3_cfg

        config_obj = botocore.config.Config(**cfg_kwargs)

        use_cache = session is None
        cache = getattr(S3Utils._RESOURCE_CACHE_LOCAL, 'cache', None)
        if use_cache and cache is None:
            cache = {}
            S3Utils._RESOURCE_CACHE_LOCAL.cache = cache

        cache_key = None
        ttl_sec = S3Utils._get_client_cache_ttl_sec()
        if use_cache and ttl_sec != 0:
            cache_key = S3Utils._make_cache_key(
                profile=profile,
                creds=creds,
                get_client=get_client,
                endpoint_url=endpoint_url,
                region_name=region_name,
                config=cfg_kwargs,
            )
            if refresh:
                cache.pop(cache_key, None)

            entry = cache.get(cache_key)
            if entry is not None:
                created_ts = entry.get('created_ts', 0.0)
                if ttl_sec < 0 or (time.monotonic() - created_ts) <= ttl_sec:
                    return entry['obj']
                cache.pop(cache_key, None)

        s3 = S3Utils._build_resource(
            profile=profile,
            creds=creds,
            session=session,
            config_obj=config_obj,
            endpoint_url=endpoint_url,
            region_name=region_name,
        )

        obj = s3.meta.client if get_client else s3

        if use_cache and cache_key is not None:
            cache[cache_key] = {'created_ts': time.monotonic(), 'obj': obj}

        return obj

    @staticmethod
    def parse_s3_url(s3_url: str) -> Optional[Tuple[str, str]]:
        match = re.match(r"s3://([^/]+)/(.*)", s3_url, flags=re.UNICODE)

        if match is None:
            return None, None

        return match.groups()[0], match.groups()[1]

    @staticmethod
    def build_s3_url(bucket, key) -> str:
        return build_s3_url(bucket, key)

    @staticmethod
    def is_s3_url(path: Optional[str]) -> bool:
        return is_s3_url(path)

    @staticmethod
    def parse_prefix_with_step(path: str) -> str:
        """
        Use regex to find the pattern up to "-step=900-"
        s3://path/to/checkpoints/tp_rank_00_pp_rank_000/megatron_gpt--step=900-validation_loss=6.47-consumed_samples=35960.0-last.ckpt
        should return s3://path/to/checkpoints/tp_rank_00_pp_rank_000/megatron_gpt--step=900-
        """
        match = re.search(r'(.*step=\d+-)', path)

        if match:
            return match.group(1)

        return path


def _scan_objects_with_retry(s3_bucket, s3_prefix):
    # this returns a collection https://boto3.amazonaws.com/v1/documentation/api/latest/guide/collections.html
    # This collection acts as an iterable that automatically makes additional requests to retrieve more objects from S3 as needed
    objects = s3_bucket.objects.filter(Prefix=s3_prefix)
    return list(objects)


def is_slow_down_error(exception):
    """
    This function checks if the error is due to slowdown or is throttling related.
    If so, returns true to allow tenacity to retry the upload/download to S3.
    """
    class_name = exception.__class__.__name__
    module_name = exception.__class__.__module__
    full_class_name = f"{module_name}.{class_name}"
    logging.error(f'Caught exception of type {full_class_name}: {exception}')

    # 2023-12-07T05:59:25.913721576Z stdout F 2023-12-07 05:59:25,913 [ERROR] - s3_utils.py:354 - Caught exception:
    # AWS_ERROR_S3_INVALID_RESPONSE_STATUS: Invalid response status from request. Body from error request is: b'<?xml version="1.0" encoding="UTF-8"?>\n<Error><Code>RequestTimeout</Code><Message>Your socket connection to the server was not read from or written to within the timeout period. Idle connections will be closed.</Message><RequestId>XPHS9896G3RJE364</RequestId><HostId>ZAiF3HPpUD5IgSr/mfkP2QPs7ttuvY+uTRG9MET/jZZ45MJ6bVbnvSBQLggICvPCROPP/1k85p4=</HostId></Error>'
    message = str(exception)
    if (
        "<Code>SlowDown</Code>" in message
        or "<Code>RequestTimeout</Code>" in message
        or "<Code>InternalError</Code>" in message
    ):
        logging.info("Identified the Retriable Error retrying the job")
        return True

    if crt_available and isinstance(exception, awscrt.exceptions.AwsCrtError):
        logging.error(f'Caught awscrt.exceptions.AwsCrtError: {exception.__repr__()}')
        return True

    if isinstance(exception, ClientError):
        logging.error(f'Caught ClientError, response is: {exception.response}')
        error_code = exception.response['Error']['Code'] if exception.response else None
        return error_code in ['SlowDown', 'RequestTimeout', 'InternalError']
    logging.info("Non Retriable Error - Terminating the job")
    return False


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_delay(2 * 60),
    retry=retry_if_exception(is_slow_down_error),
    before_sleep=before_sleep_log(logging, logging.ERROR),
)
def _download_fileobj_with_retry(
    s3_client, bucket: str, key: str, bytes_buffer: BytesIO, config: TransferConfig = None
):
    s3_client.download_fileobj(bucket, key, bytes_buffer, Config=config)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_delay(2 * 60),
    retry=retry_if_exception(is_slow_down_error),
    before_sleep=before_sleep_log(logging, logging.ERROR),
)
def _download_file_with_retry(s3_client, bucket: str, key: str, file_path: str, config: TransferConfig = None):
    s3_client.download_file(bucket, key, file_path, Config=config)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_delay(2 * 60),
    retry=retry_if_exception(is_slow_down_error),
    before_sleep=before_sleep_log(logging, logging.ERROR),
)
def _upload_fileobj_with_retry(s3_client, bytes_buffer: BytesIO, bucket: str, key: str, config: TransferConfig = None):
    s3_client.upload_fileobj(bytes_buffer, bucket, key, Config=config)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_delay(2 * 60),
    retry=retry_if_exception(is_slow_down_error),
    before_sleep=before_sleep_log(logging, logging.ERROR),
)
def _upload_file_with_retry(s3_client, file_path: str, bucket: str, key: str, config: TransferConfig = None):
    s3_client.upload_file(file_path, bucket, key, Config=config)
