"""Download the prebuilt sysimage for this platform and Julia environment.

CI publishes one image per platform to the GitHub release tagged
``sysimage-<key>``, where the key identifies the exact Julia environment.
"""

import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _load import sysimage  # noqa: E402


def main() -> None:
    key = sysimage.current_key()
    image = sysimage.image_path(key)
    if image.is_file():
        print(f"[sysimage] already present: {image}")
        return

    url = f"{sysimage.RELEASES_URL}/{sysimage.release_tag(key)}/{image.name}"
    print(f"[sysimage] downloading {url}", flush=True)
    image.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(dir=image.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as resp:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        os.replace(partial, image)
    # never fail: other tasks depend on this one, and galley works without an
    # image, just with a slower first call
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"[sysimage] download failed ({e}); continuing without it.")
            return
        print(
            f"[sysimage] no prebuilt image for this platform and Julia environment "
            f"({image.name}); continuing without it. Build one locally with "
            "`pixi run build-sysimage`."
        )
        return
    except (urllib.error.URLError, OSError) as e:
        print(f"[sysimage] download failed ({e}); continuing without it.")
        return
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    print(f"[sysimage] saved {image}")


if __name__ == "__main__":
    main()
