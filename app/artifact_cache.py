"""Signature-keyed display verification; research evaluation always hashes afresh."""
import json
from pathlib import Path
from typing import Any

import streamlit as st

from src.utils.cohort import verify_cohort
from src.utils.integrity import sha256_file

FileSignature = tuple[str, int, int, int, int, int]


def file_signature(path: str | Path) -> FileSignature:
    path = Path(path).resolve()
    stamp = path.stat()
    return (str(path), stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns)


@st.cache_data(show_spinner=False, max_entries=4096)
def _file_hash(signature: FileSignature) -> str:
    result = sha256_file(signature[0])
    if file_signature(signature[0]) != signature:
        raise ValueError('Artifact changed during verification; refresh the experiment')
    return result


def verified_hash(path: str | Path) -> str:
    return _file_hash(file_signature(path))


@st.cache_data(show_spinner=False, max_entries=256)
def _json_document(signature: FileSignature) -> Any:
    value = json.loads(Path(signature[0]).read_text(encoding='utf-8'))
    if file_signature(signature[0]) != signature:
        raise ValueError('Artifact changed while reading its manifest')
    return value


def read_json(path: str | Path) -> Any:
    return _json_document(file_signature(path))


def dataset_fingerprint(root: str | Path) -> str:
    # Metadata is small and read on every rerun. Every referenced tensor is stat'ed,
    # but unchanged tensor bytes are not reread merely to move a slice slider.
    return verify_cohort(root, hash_file=verified_hash).fingerprint


def clear_verification_cache() -> None:
    _file_hash.clear()
    _json_document.clear()
