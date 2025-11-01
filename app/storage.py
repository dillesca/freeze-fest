from __future__ import annotations

import os
from typing import BinaryIO

GCS_PHOTO_BUCKET = os.getenv("GCS_PHOTO_BUCKET")
GCS_PHOTO_BASE_URL = os.getenv("GCS_PHOTO_BASE_URL")
GCS_PHOTO_CACHE_CONTROL = os.getenv("GCS_PHOTO_CACHE_CONTROL", "public, max-age=86400")

_GCS_IDENTIFIER_PREFIX = "gcs:"


def gcs_photos_enabled() -> bool:
    """Return True when a Google Cloud Storage bucket is configured for photos."""
    return bool(GCS_PHOTO_BUCKET)


def make_gcs_identifier(object_name: str) -> str:
    """Encode an object name so we can store it in the Photo table."""
    return f"{_GCS_IDENTIFIER_PREFIX}{object_name.lstrip('/')}"


def is_gcs_identifier(value: str) -> bool:
    return value.startswith(_GCS_IDENTIFIER_PREFIX)


def extract_object_name(identifier: str) -> str:
    if not is_gcs_identifier(identifier):
        raise ValueError("Identifier does not reference GCS content.")
    return identifier[len(_GCS_IDENTIFIER_PREFIX) :].lstrip("/")


def gcs_public_url(object_name: str) -> str:
    if not GCS_PHOTO_BUCKET:
        raise RuntimeError("GCS_PHOTO_BUCKET is not configured.")
    base_url = (GCS_PHOTO_BASE_URL or f"https://storage.googleapis.com/{GCS_PHOTO_BUCKET}").rstrip("/")
    return f"{base_url}/{object_name.lstrip('/')}"


def upload_photo_stream(handle: BinaryIO, *, object_name: str, content_type: str) -> str:
    """Upload the provided file-like object to the configured GCS bucket."""
    if not gcs_photos_enabled():
        raise RuntimeError("GCS photo storage is not enabled.")

    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - dependency absent in some envs
        raise RuntimeError(
            "google-cloud-storage is required to upload photos to GCS."
        ) from exc

    client = storage.Client()
    bucket = client.bucket(GCS_PHOTO_BUCKET)
    blob = bucket.blob(object_name.lstrip("/"))
    blob.upload_from_file(handle, content_type=content_type)
    if GCS_PHOTO_CACHE_CONTROL:
        blob.cache_control = GCS_PHOTO_CACHE_CONTROL
        blob.patch()
    return gcs_public_url(object_name)

