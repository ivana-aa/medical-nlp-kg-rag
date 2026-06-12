"""Download CMeIE JSONL files from Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote

import requests


CMeIE_DATASET_REPO = "Aunderline/CMeIE"
CMeIE_REVISION = "main"
CMeIE_HF_FILES = (
    "53_schemas.jsonl",
    "CMeIE_train.jsonl",
    "CMeIE_dev.jsonl",
    "CMeIE_test.jsonl",
)


def build_hf_resolve_url(filename: str, repo_id: str = CMeIE_DATASET_REPO, revision: str = CMeIE_REVISION) -> str:
    """Build a direct Hugging Face Hub URL for one dataset file."""
    safe_filename = quote(filename.strip("/"))
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{safe_filename}"


def download_file(
    url: str,
    output_path: str | Path,
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,
    session: Callable[..., object] | None = None,
) -> Path:
    """Download one URL to a file through a temporary path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    getter = session or requests.get
    response = getter(url, stream=True, timeout=timeout)
    response.raise_for_status()

    with temp_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    temp_path.replace(output_path)
    return output_path


def download_cmeie_files(
    output_dir: str | Path,
    files: Iterable[str] = CMeIE_HF_FILES,
    skip_existing: bool = True,
    timeout: int = 60,
    session_get: Callable[..., object] | None = None,
) -> list[Path]:
    """Download selected CMeIE files into a local raw-data directory."""
    output_dir = Path(output_dir)
    downloaded_paths: list[Path] = []
    for filename in files:
        output_path = output_dir / filename
        if skip_existing and output_path.exists():
            downloaded_paths.append(output_path)
            continue

        url = build_hf_resolve_url(filename)
        downloaded_paths.append(
            download_file(
                url,
                output_path,
                timeout=timeout,
                session=session_get,
            )
        )
    return downloaded_paths
