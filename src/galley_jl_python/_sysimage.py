"""Locate a prebuilt Julia sysimage and point juliacall at it.

This module must not import juliacall: it runs before Julia starts.

A sysimage only works with the exact Julia version and package versions it was
built from, so each image is named after a hash of both (see ``env_key``) and
lives in a per-user cache directory. An image is used only when its name
matches the current environment; otherwise Julia starts normally.

Get an image with ``pixi run fetch-sysimage`` (downloads the one CI built for
this environment) or ``pixi run build-sysimage`` (builds it locally). Set
``GALLEY_JL_PYTHON_SYSIMAGE=0`` to never load one, or
``PYTHON_JULIACALL_SYSIMAGE`` to use a specific image.
"""

import hashlib
import json
import os
import platform
import sys
import tomllib
from pathlib import Path

RELEASES_URL = "https://github.com/finch-tensor/galley-jl-python/releases/download"
_EXTENSIONS = {"linux": "so", "darwin": "dylib", "win32": "dll"}


def cache_dir() -> Path:
    if custom := os.environ.get("GALLEY_JL_PYTHON_CACHE"):
        return Path(custom)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "galley-jl-python"


def env_key(project: str) -> str:
    """Hash the Julia version and resolved package set of a juliapkg project."""
    manifest = tomllib.loads((Path(project) / "Manifest.toml").read_text())
    resolved = sorted(
        (name, entry.get("uuid"), entry.get("version"), entry.get("git-tree-sha1"))
        for name, entries in manifest["deps"].items()
        for entry in entries
    )
    payload = json.dumps([manifest["julia_version"], resolved])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def release_tag(key: str) -> str:
    return f"sysimage-{key}"


def image_name(key: str) -> str:
    ext = _EXTENSIONS.get(sys.platform, "so")
    return f"galley-{sys.platform}-{platform.machine().lower()}-{key}.{ext}"


def image_path(key: str) -> Path:
    return cache_dir() / image_name(key)


def current_key() -> str:
    import juliapkg

    return env_key(juliapkg.project())


def configure() -> None:
    """Point juliacall at the cached sysimage for this environment, if any."""
    if os.environ.get("GALLEY_JL_PYTHON_SYSIMAGE") == "0":
        return
    if "PYTHON_JULIACALL_SYSIMAGE" in os.environ:
        return
    image = image_path(current_key())
    if image.is_file():
        os.environ["PYTHON_JULIACALL_SYSIMAGE"] = str(image)
