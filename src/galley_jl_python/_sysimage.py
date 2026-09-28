"""Locate, and if needed download, a prebuilt Julia sysimage for juliacall.

This module must not import juliacall: it runs before Julia starts.

A sysimage only works with the exact Julia version and package versions it was
built from, so each image is named after a hash of both (see ``env_key``) and
lives in a per-user cache directory. An image is used only when its name
matches the current environment; otherwise Julia starts normally.

On import, galley downloads the image CI published for this platform and
environment if it isn't cached yet, and uses it right away. Set
``GALLEY_JL_PYTHON_SYSIMAGE=0`` to never download or load one, or
``PYTHON_JULIACALL_SYSIMAGE`` to use a specific image. In a galley checkout,
``pixi run build-sysimage`` builds one locally.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

RELEASES_URL = "https://github.com/finch-tensor/galley-jl-python/releases/download"
# Bump when images built from the same environment become incompatible, so the
# new images get new names instead of reusing the published ones.
# 2: PythonCall finds juliacall from Python (v1 only worked on the build host)
IMAGE_FORMAT = 2
_EXTENSIONS = {"linux": "so", "darwin": "dylib", "win32": "dll"}
# after a 404, don't ask again for this long (images are published after CI)
_MISSING_TTL = 24 * 60 * 60
# a lock older than this is left over from a download that was killed
_STALE_LOCK = 2 * 60 * 60


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
    arch = platform.machine().lower()
    return f"galley-v{IMAGE_FORMAT}-{sys.platform}-{arch}-{key}.{ext}"


def image_path(key: str) -> Path:
    return cache_dir() / image_name(key)


def current_key() -> str:
    import juliapkg

    return env_key(juliapkg.project())


def _log(message: str) -> None:
    print(f"[galley-jl-python] {message}", file=sys.stderr, flush=True)


def _acquire_lock(lock: Path) -> bool:
    for _ in range(2):
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime < _STALE_LOCK:
                    return False
                lock.unlink()
            except FileNotFoundError:
                pass
    return False


def _download(url: str, dest: Path, attempts: int = 4) -> None:
    """Download ``url`` to ``dest``, resuming after a dropped connection."""
    size = done = 0
    next_report = 0.25
    for attempt in range(attempts):
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            with (
                urllib.request.urlopen(req, timeout=60) as resp,
                dest.open("ab") as out,
            ):
                if done and resp.status != 206:  # server ignored the range
                    out.truncate(0)
                    done = 0
                if not size:
                    size = done + int(resp.headers.get("Content-Length") or 0)
                    _log(
                        f"downloading the prebuilt Julia sysimage ({size >> 20} MB), "
                        "once per environment; set GALLEY_JL_PYTHON_SYSIMAGE=0 to skip"
                    )
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if size and done / size >= next_report:
                        _log(f"  {done * 100 // size}%")
                        next_report += 0.25
            if not size or done >= size:
                return
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError):
            if attempt == attempts - 1:
                raise
        _log(
            f"  connection dropped at {done >> 20} MB; resuming"
            if done
            else "  couldn't connect; retrying"
        )


def _works(image: Path) -> bool:
    """Check that juliacall starts with ``image``, in a throwaway process."""
    env = {**os.environ, "PYTHON_JULIACALL_SYSIMAGE": str(image)}
    env["GALLEY_JL_PYTHON_SYSIMAGE"] = "0"
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from juliacall import Main; Main.seval('1 + 1')"],
            env=env,
            capture_output=True,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def fetch(key: str) -> Path | None:
    """Download the published image for ``key`` into the cache.

    Returns the image path, or None if there is no image or it couldn't be
    downloaded. Never raises: galley works without an image, just with a slow
    first call. If another process is already downloading, returns None
    immediately instead of waiting.
    """
    image = image_path(key)
    if image.is_file():
        return image
    missing = image.with_name(image.name + ".missing")
    lock = image.with_name(image.name + ".lock")
    try:
        if missing.is_file() and time.time() - missing.stat().st_mtime < _MISSING_TTL:
            return None
        image.parent.mkdir(parents=True, exist_ok=True)
        if not _acquire_lock(lock):
            return None
    except OSError:
        return None

    url = f"{RELEASES_URL}/{release_tag(key)}/{image.name}"
    partial = None
    try:
        fd, partial = tempfile.mkstemp(dir=image.parent, suffix=".part")
        os.close(fd)
        _download(url, Path(partial))
        if not _works(Path(partial)):
            _log("the downloaded sysimage doesn't load here; starting without it")
            missing.touch()
            return None
        os.chmod(partial, 0o644)
        os.replace(partial, image)
        partial = None
        _log(f"saved {image}")
        return image
    except urllib.error.HTTPError as e:
        if e.code == 404:
            missing.touch()
        else:
            _log(f"sysimage download failed ({e}); starting without it")
        return None
    except (urllib.error.URLError, OSError) as e:
        _log(f"sysimage download failed ({e}); starting without it")
        return None
    finally:
        if partial is not None and os.path.exists(partial):
            os.remove(partial)
        lock.unlink(missing_ok=True)


def configure() -> None:
    """Point juliacall at this environment's sysimage, downloading it if needed."""
    if os.environ.get("GALLEY_JL_PYTHON_SYSIMAGE") == "0":
        return
    if "PYTHON_JULIACALL_SYSIMAGE" in os.environ:
        return
    image = fetch(current_key())
    if image is not None:
        os.environ["PYTHON_JULIACALL_SYSIMAGE"] = str(image)
