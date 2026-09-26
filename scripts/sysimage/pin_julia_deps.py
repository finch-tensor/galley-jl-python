"""Pin galley's Julia dependencies to the currently resolved environment.

Rewrites ``src/galley_jl_python/juliapkg.json`` so that Julia and every
registered package in the resolved ``Manifest.toml`` (transitive dependencies
included) are pinned to exact versions. Everyone then resolves the same
manifest, which the prebuilt sysimage requires.

To update: loosen the pins you want to move, resolve (e.g. ``pixi run
compile``), run this script, then rebuild with ``pixi run build-sysimage``.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JULIAPKG_JSON = ROOT / "src" / "galley_jl_python" / "juliapkg.json"
# juliacall pins these itself (PythonCall to its own release, OpenSSL_jll to
# match Python's OpenSSL), so they must stay unpinned here
UNPINNED = {"OpenSSL_jll", "PythonCall"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(sys.prefix) / "julia_env" / "Manifest.toml",
    )
    args = parser.parse_args()

    manifest = tomllib.loads(args.manifest.read_text())
    config = json.loads(JULIAPKG_JSON.read_text())
    config["julia"] = f"={manifest['julia_version']}"

    packages = {}
    for name, (entry, *_) in sorted(manifest["deps"].items()):
        # stdlibs have no tree hash and ship with Julia itself
        if "git-tree-sha1" not in entry or name in UNPINNED:
            continue
        # drop build metadata (e.g. JLL "+0"): juliapkg can't parse it and Pkg
        # compat ignores it anyway
        version = entry["version"].split("+")[0]
        packages[name] = {"uuid": entry["uuid"], "version": f"={version}"}
    config["packages"] = packages

    JULIAPKG_JSON.write_text(json.dumps(config, indent=2) + "\n")
    print(f"pinned julia {config['julia']} and {len(packages)} packages")


if __name__ == "__main__":
    main()
