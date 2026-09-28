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

Galley needs a Python linked against **OpenSSL 3.5 or newer**. It runs Julia
1.12, which bundles OpenSSL 3.5 in the same process as Python. Pythons from
conda-forge, including through pixi or conda, meet this. Many system and
pyenv-built Pythons still use OpenSSL 3.0, and there `import galley_jl_python`
fails with an error that says so. Check with
`python -c "import ssl; print(ssl.OPENSSL_VERSION)"`.

## Contributing

### Development environment

Galley uses [pixi](https://pixi.sh) for development. All configuration lives in
`pyproject.toml`: the package metadata and dependencies are under `[project]`,
and pixi's settings are under `[tool.pixi]`.

To set up, clone the repository and run:
```bash
pixi install          # the default environment
pixi install -e test  # adds the test dependencies (the `test` extra)
```

pixi installs the package in editable mode. Julia is not a pixi dependency:
[juliapkg](https://github.com/JuliaPy/pyjuliapkg) installs the pinned Julia
version and packages the first time `galley_jl_python` is imported. To trigger
that, and to fetch the sysimage (see below), run:
```bash
pixi run compile
```

Run any other command inside an environment with `pixi run`, for example
`pixi run -e test python`.

The package is still built and published with Poetry (see
[Publishing](#publishing)).

### Working with a local copy of Finch.jl
The `develop.py` script can be used to set up a local copy of Finch.jl for
development. Run it with `pixi run python develop.py`.

```
Usage:
    develop.py [--restore] [--path <path>]

Options:
    --restore   Restore the original juliapkg.json file.
    --path      Path to the local copy of Finch.jl [default: ../Finch.jl].
```

### Julia sysimage

Most of Galley's startup time is Julia compiling Finch itself. A prebuilt Julia
sysimage removes it: the first matmul of a session drops from about 4.5 minutes
to a few seconds.

- `import galley_jl_python` uses the image for the current platform and Julia
  environment. If it isn't cached in `~/.cache/galley-jl-python/`
  (`GALLEY_JL_PYTHON_CACHE` overrides this), the import downloads it from the
  project's GitHub releases first, about 900 MB once per environment. It then
  checks that the image loads, and if there's no image or anything fails, Julia
  starts without it. Set `GALLEY_JL_PYTHON_SYSIMAGE=0` to turn all of this off.
- `pixi run fetch-sysimage` downloads the image ahead of time. `pixi run compile`
  and `pixi run test` run this step first.
- `pixi run build-sysimage` builds the image locally instead, in about half an
  hour.

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
GitHub release, where imports and `fetch-sysimage` find them. The action can also be run
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

The hooks run in their own `pre-commit` environment, which doesn't install
galley's dependencies or Julia.

```bash
pixi run pre-commit-install   # run the hooks on every `git commit`
pixi run pre-commit           # run every hook on every file now
pixi run pre-commit ruff      # the same, skipping the listed hook ids
```

To run a single hook, use `pixi shell -e pre-commit` and then
`pre-commit run <hook-id> -a`.

### Testing

Galley uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run the
tests:

```bash
pixi run test      # one pytest-xdist worker per CPU
pixi run test 4    # or a fixed number of workers
```

This runs `compile` first, then two suites one after the other, each spread
over the workers:

- `pixi run test-unit` runs the unit tests.
- `pixi run test-array-api` runs the Array API tests described below.

Each worker is a separate Julia process that uses 1–2 GB of memory, so lower
the worker count on machines with little memory.

To run a subset, call pytest in the test environment directly:

```bash
pixi run -e test pytest tests/test_fused.py
```

Array API tests are included in `tests/test_array_api.py`. These tests invoke
the [Array API Conformance Tests](https://github.com/data-apis/array-api-tests).
To forward `pytest` options to the nested
`array-api-tests` invocation, use `--array-api` (alias:
`--array-api-pytest-args`):

```bash
pixi run -e test pytest tests/test_array_api.py \
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
