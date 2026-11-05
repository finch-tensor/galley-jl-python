#!/usr/bin/env poetry run python
import argparse
import os
import shutil

import juliapkg

script_dir = os.path.dirname(os.path.abspath(__file__))
source_file = os.path.join(script_dir, "src/galley_jl_python/juliapkg.json")
backup_file = os.path.join(script_dir, "src/galley_jl_python/juliapkg.json.orig")

# Parse command-line arguments
usage = """
Usage:
    develop.py [--restore] [--path <path>]

Options:
    --restore   Restore the original juliapkg.json file.
    --path      Path to the local copy of Galley.jl [default: ../Galley.jl].
"""
parser = argparse.ArgumentParser(
    description=(
        "Development script for Galley. This script allows you to specify "
        "the location of a local copy of Galley.jl."
    ),
    usage=usage,
)
parser.add_argument(
    "--path",
    default=os.path.join(script_dir, "../Galley.jl"),
    help="Path to the Galley.jl package.",
)
parser.add_argument(
    "--restore", action="store_true", help="Restore the original juliapkg.json file."
)
args = parser.parse_args()

# Handle the --restore flag
if args.restore:
    try:
        shutil.copy(backup_file, source_file)
        print("Restored src/galley_jl_python/juliapkg.json from backup.")
    except FileNotFoundError:
        print(
            "Error: Backup file src/galley_jl_python/juliapkg.json.orig does not exist."
        )
    except (OSError, PermissionError) as e:
        print(f"An error occurred: {e}")
    exit()

# Set the Galley path
galley_path = os.path.abspath(args.path)

# Define source and destination file paths and copy the file
try:
    if not os.path.exists(backup_file):
        shutil.copy(source_file, backup_file)
except (OSError, PermissionError) as e:
    print(f"An error occurred: {e}")

# Check out Galley for development

juliapkg.rm("Galley", target="src/galley_jl_python/juliapkg.json")
juliapkg.add(
    "Galley",
    "0b99cc74-af0a-4786-be99-594ac14242a4",
    dev=True,
    path=galley_path,
    target="src/galley_jl_python/juliapkg.json",
)
