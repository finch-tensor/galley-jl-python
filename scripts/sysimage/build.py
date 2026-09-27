"""Build a Julia sysimage that removes most of galley's first-call compile time.

Traces the Julia methods compiled by ``workload.py``, compiles them into a
sysimage with PackageCompiler, and stores it in the per-user cache under a name
tied to the current Julia environment, where ``galley_jl_python`` finds it.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import juliapkg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _load import sysimage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--statements",
        type=Path,
        help="reuse precompile statements from an earlier trace instead of tracing",
    )
    args = parser.parse_args()

    project = juliapkg.project()
    key = sysimage.env_key(project)
    image = sysimage.image_path(key)
    image.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        statements = args.statements or Path(tmp) / "precompile_statements.jl"
        if args.statements is None:
            print(f"[sysimage] tracing workload -> {statements}", flush=True)
            env = {
                k: v for k, v in os.environ.items() if k != "PYTHON_JULIACALL_SYSIMAGE"
            }
            subprocess.run(
                [sys.executable, str(HERE / "workload.py")],
                env={
                    **env,
                    "GALLEY_JL_PYTHON_SYSIMAGE": "0",
                    "PYTHON_JULIACALL_TRACE_COMPILE": str(statements),
                },
                check=True,
            )

        # build next to the target and rename, so an interrupted build never
        # leaves a partial image where it would be loaded
        partial = Path(tmp) / image.name
        print(f"[sysimage] building {image.name} (this takes a while)", flush=True)
        subprocess.run(
            [
                juliapkg.executable(),
                "--startup-file=no",
                str(HERE / "build_sysimage.jl"),
                project,
                str(statements),
                str(partial),
            ],
            check=True,
        )
        shutil.move(partial, image)

    print(f"[sysimage] wrote {image}", flush=True)
    if github_output := os.environ.get("GITHUB_OUTPUT"):
        with open(github_output, "a") as f:
            f.write(f"key={key}\npath={image}\ntag={sysimage.release_tag(key)}\n")


if __name__ == "__main__":
    main()
