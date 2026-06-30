# -*- coding: utf-8 -*-
"""Shared model resolution helpers for startup checks and runtime loaders."""
from __future__ import annotations

import os
from typing import Iterable, Optional

from app.config.defaults import (
    MANGA_OCR_FILES,
    PADDLE_OCR_VL_MMPROJ_FILE,
    PADDLE_OCR_VL_MODEL_FILE,
    PADDLE_OCR_VL_REPO_ID,
)


def _app_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def models_root() -> str:
    return os.path.join(_app_root(), "models")


def _hf_home() -> str:
    return os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )


def _hf_snapshot_dirs(user: str, repo: str) -> list[str]:
    model_dir = os.path.join(_hf_home(), "hub", f"models--{user}--{repo}", "snapshots")
    if not os.path.isdir(model_dir):
        return []
    dirs: list[str] = []
    for entry in os.listdir(model_dir):
        path = os.path.join(model_dir, entry)
        if os.path.isdir(path):
            dirs.append(path)
    return dirs


def _first_dir_with_files(dirs: Iterable[str], required_files: Iterable[str]) -> Optional[str]:
    required = tuple(required_files)
    for directory in dirs:
        if all(os.path.exists(os.path.join(directory, name)) for name in required):
            return directory
    return None

def resolve_manga_ocr_system_ref() -> Optional[str]:
    snapshot = _first_dir_with_files(
        _hf_snapshot_dirs("kha-white", "manga-ocr-base"),
        MANGA_OCR_FILES,
    )
    if snapshot:
        return snapshot
    return None


def resolve_manga_ocr_local_dir(base_dir: Optional[str] = None) -> Optional[str]:
    target = os.path.join(base_dir or models_root(), "manga-ocr")
    if _first_dir_with_files([target], MANGA_OCR_FILES):
        return target
    return None


def _paddle_ocr_vl_local_dir(base_dir: Optional[str] = None) -> str:
    return os.path.join(base_dir or models_root(), "paddleocr-vl-1.6-gguf")


def _paddle_ocr_vl_snapshot_dirs() -> list[str]:
    user, repo = PADDLE_OCR_VL_REPO_ID.split("/", 1)
    return _hf_snapshot_dirs(user, repo)


def resolve_paddle_ocr_vl_model_file(base_dir: Optional[str] = None) -> Optional[str]:
    override = os.environ.get("MT_PADDLEOCR_VL_MODEL")
    if override and os.path.isfile(override):
        return override
    local_path = os.path.join(_paddle_ocr_vl_local_dir(base_dir), PADDLE_OCR_VL_MODEL_FILE)
    if os.path.isfile(local_path):
        return local_path
    snapshot = _first_dir_with_files(_paddle_ocr_vl_snapshot_dirs(), [PADDLE_OCR_VL_MODEL_FILE])
    if snapshot:
        return os.path.join(snapshot, PADDLE_OCR_VL_MODEL_FILE)
    return None


def resolve_paddle_ocr_vl_mmproj_file(base_dir: Optional[str] = None) -> Optional[str]:
    override = os.environ.get("MT_PADDLEOCR_VL_MMPROJ")
    if override and os.path.isfile(override):
        return override
    local_path = os.path.join(_paddle_ocr_vl_local_dir(base_dir), PADDLE_OCR_VL_MMPROJ_FILE)
    if os.path.isfile(local_path):
        return local_path
    snapshot = _first_dir_with_files(_paddle_ocr_vl_snapshot_dirs(), [PADDLE_OCR_VL_MMPROJ_FILE])
    if snapshot:
        return os.path.join(snapshot, PADDLE_OCR_VL_MMPROJ_FILE)
    return None


def resolve_llama_server_executable(base_dir: Optional[str] = None) -> Optional[str]:
    override = os.environ.get("MT_PADDLEOCR_VL_LLAMA_SERVER")
    if override and os.path.isfile(override):
        return override
    root = os.path.join(base_dir or models_root(), "llama.cpp")
    if not os.path.isdir(root):
        return None
    candidates: list[str] = []
    for current_root, _dirs, files in os.walk(root):
        for name in files:
            if name.lower() == "llama-server.exe":
                candidates.append(os.path.join(current_root, name))
    if not candidates:
        return None
    candidates.sort(key=lambda path: ("cuda" not in path.lower(), len(path), path.lower()))
    return candidates[0]


def has_paddle_ocr_vl_runtime(base_dir: Optional[str] = None) -> bool:
    return bool(
        resolve_paddle_ocr_vl_model_file(base_dir)
        and resolve_paddle_ocr_vl_mmproj_file(base_dir)
        and resolve_llama_server_executable(base_dir)
    )


def resolve_ner_system_snapshot() -> Optional[str]:
    required = ("config.json",)
    optional_weights = ("model.safetensors", "pytorch_model.bin")
    for snapshot in _hf_snapshot_dirs("jurabi", "bert-ner-japanese"):
        if not all(os.path.exists(os.path.join(snapshot, name)) for name in required):
            continue
        if any(os.path.exists(os.path.join(snapshot, name)) for name in optional_weights):
            return snapshot
    return None


def resolve_ner_local_dir(model_dir: Optional[str] = None) -> Optional[str]:
    base_dir = model_dir or os.path.join(models_root(), "ner")
    candidates = [base_dir]
    nested = os.path.join(base_dir, "models--jurabi--bert-ner-japanese", "snapshots")
    if os.path.isdir(nested):
        for entry in os.listdir(nested):
            path = os.path.join(nested, entry)
            if os.path.isdir(path):
                candidates.append(path)

    required = ("config.json",)
    optional_weights = ("model.safetensors", "pytorch_model.bin")
    for candidate in candidates:
        if not all(os.path.exists(os.path.join(candidate, name)) for name in required):
            continue
        if any(os.path.exists(os.path.join(candidate, name)) for name in optional_weights):
            return candidate
    return None
