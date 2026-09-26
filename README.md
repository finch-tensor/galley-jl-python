# galley-jl-python

This is the beginnings of a sparse tensor library for Python, backed by the
[Finch.jl](https://github.com/finch-tensor/Finch.jl) tensor compiler.

## Source

The source code for `galley-jl-python` is available on GitHub at [https://github.com/finch-tensor/galley-jl-python](https://github.com/finch-tensor/galley-jl-python).

## Installation

`galley-jl-python` is available on PyPI, and can be installed with pip:
```bash
pip install galley-jl-python
```

## Contributing

### Packaging

Galley uses [poetry](https://python-poetry.org/) for packaging.

To install for development, clone the repository and run:
```bash
poetry install --with test
```
to install the current project and dev dependencies.

### Working with a local copy of Finch.jl
The `develop.py ` script can be used to set up a local copy of Finch.jl for development.

```
Usage:
    develop.py [--restore] [--path <path>]

Options:
    --restore   Restore the original juliapkg.json file.
    --path      Path to the local copy of Finch.jl [default: ../Finch.jl].
```

### Julia sysimage

Most of Galley's startup time is Julia compiling Finch itself. A prebuilt Julia
sysimage removes it: the first operations of a session drop from minutes to a
few seconds.

- `pixi run fetch-sysimage` downloads the image for your platform into
  `~/.cache/galley-jl-python/` (`GALLEY_JL_PYTHON_CACHE` overrides this).
  `pixi run compile` and `pixi run test` run this step first.
- `pixi run build-sysimage` builds the image locally instead. This takes hours.
- `import galley_jl_python` loads a cached image automatically when it matches
  the current Julia environment. Otherwise Julia starts without it. Set
  `GALLEY_JL_PYTHON_SYSIMAGE=0` to turn this off.

An image works only with the exact Julia and package versions it was built
from. For this reason `src/galley_jl_python/juliapkg.json` pins Julia and every
Julia package. Each image's name includes a hash of that environment. A local
Finch.jl from `develop.py` therefore runs without the image.

To update the Julia dependencies:

1. Loosen the pins you want to change.
2. Resolve with `pixi run compile`.
3. Re-pin with `python scripts/sysimage/pin_julia_deps.py`.

Pushing the new pins to `main` runs the "Sysimage" GitHub Action. It builds
images for Linux, macOS and Windows and publishes them to a `sysimage-<hash>`
GitHub release, where `fetch-sysimage` finds them. The action can also be run
manually from the Actions tab.

### Publishing

The "Publish" GitHub Action is a manual workflow for publishing Python packages to PyPI using Poetry. It handles the version management based on the `pyproject.toml` file and automates tagging and creating GitHub releases.

#### Version Update

Before initiating the "Publish" action, update the package's version number in `pyproject.toml`. Follow semantic versioning guidelines for this update.

#### Triggering the Action

The action is triggered manually. Once the version in `pyproject.toml` is updated, manually start the "Publish" action from the GitHub repository's Actions tab.

#### Process and Outcomes

On successful execution, the action publishes the package to PyPI and tags the release in the GitHub repository. If the version number is not updated, the action fails to publish to PyPI, and no tagging or release is done. In case of failure, correct the version number and rerun the action.

#### Best Practices

- Ensure the version number in `pyproject.toml` is updated before triggering the action.
- Regularly check action logs for successful completion or to identify issues.

### Pre-commit hooks

To add pre-commit hooks, run:
```bash
poetry run pre-commit install
```

### Testing

Finch uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run the
tests:

```bash
poetry run pytest
```

Array API tests are included in `tests/test_array_api.py`. These tests invoke
the [Array API Conformance Tests](https://github.com/data-apis/array-api-tests).
To forward `pytest` options to the nested
`array-api-tests` invocation, use `--array-api` (alias:
`--array-api-pytest-args`):

```bash
poetry run pytest tests/test_array_api.py \
    --array-api="-k creation_functions" \
    --array-api="-x"
```

By default, the nested Array API run forwards common top-level pytest options
from your main invocation:

- `-x`/`--maxfail`
- `-s`
- `-v`, `-vv`, etc.
- `-k`

You can repeat `--array-api` (or `--array-api-pytest-args`) multiple times.
Each value is parsed like shell arguments and appended to the nested `pytest`
call.

`ARRAY_API_TESTS_ARGS` is still supported as a fallback for compatibility, but
the CLI option is preferred.
