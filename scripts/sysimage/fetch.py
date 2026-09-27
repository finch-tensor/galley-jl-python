"""Download the prebuilt sysimage for this platform and Julia environment.

Importing galley does this automatically; this script does it ahead of time.
CI publishes one image per platform to the GitHub release tagged
``sysimage-<key>``, where the key identifies the exact Julia environment.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _load import sysimage  # noqa: E402


def main() -> None:
    key = sysimage.current_key()
    image = sysimage.fetch(key)
    if image is not None:
        print(f"[sysimage] {image}")
    else:
        print(
            f"[sysimage] no prebuilt image available for {sysimage.image_name(key)}"
            " (or another process is downloading it); galley works without it. "
            "Build one locally with `pixi run build-sysimage`."
        )


if __name__ == "__main__":
    main()
