"""Shared utilities used across pipeline modules."""

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def download_pdf(url: str, output_path: Path) -> bool:
    """
    Download a PDF from a URL and save it to disk.

    Args:
        url: PDF URL
        output_path: Path to save the PDF

    Returns:
        True if download was successful, False otherwise.
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Downloaded PDF to %s", output_path)
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("Error downloading PDF from %s: %s", url, exc)
        return False
