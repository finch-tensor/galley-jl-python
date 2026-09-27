"""Print the sysimage key, release tag and asset name for this environment.

In GitHub Actions they are also written to ``$GITHUB_OUTPUT`` as ``key``,
``tag`` and ``name``.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _load import sysimage  # noqa: E402


def main() -> None:
    key = sysimage.current_key()
    info = {
        "key": key,
        "tag": sysimage.release_tag(key),
        "name": sysimage.image_name(key),
    }
    for k, v in info.items():
        print(f"{k}={v}")
    if github_output := os.environ.get("GITHUB_OUTPUT"):
        with open(github_output, "a") as f:
            f.writelines(f"{k}={v}\n" for k, v in info.items())


if __name__ == "__main__":
    main()
