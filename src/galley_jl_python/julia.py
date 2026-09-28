import ssl  # noqa: I001

# Julia 1.12, which galley pins, bundles OpenSSL 3.5, and Python and Julia share
# one process, so juliapkg only allows Julia 1.12 with a Python linked against
# OpenSSL 3.5 or newer. Without this check the failure is juliapkg finding no
# compatible Julia version.
if ssl.OPENSSL_VERSION_INFO[:2] < (3, 5):
    raise ImportError(
        "galley-jl-python needs a Python linked against OpenSSL 3.5 or newer, but "
        f"this one uses {ssl.OPENSSL_VERSION}. Galley runs Julia 1.12, which "
        "bundles OpenSSL 3.5 and shares this process with Python. Use a Python "
        "from conda-forge, e.g. through pixi or conda, which ships a current "
        "OpenSSL."
    )

import juliapkg  # noqa: E402, F401

from . import _sysimage  # noqa: E402

_sysimage.configure()

# To change the version of Finch used, see the documentation for pyjuliapkg here: https://github.com/JuliaPy/pyjuliapkg
# Use pyjuliapkg to modify the `juliapkg.json` file in the root of this repo.
# You can also run `develop.py` to quickly use a local copy of Finch.jl.
# An example development json is found in `juliapkg_dev.json`
# Julia and all packages are pinned so the prebuilt sysimage matches; update the
# pins with `scripts/sysimage/pin_julia_deps.py`. Finch is pinned to a git
# commit (`rev`), which that script leaves alone.
import juliacall as jc  # noqa: E402, F401
from juliacall import Main as jl  # noqa: E402, F401

jl.seval("using Finch")
jl.seval("using HDF5")
jl.seval("using NPZ")
jl.seval("using TensorMarket")
jl.seval("using Random")
jl.seval("using Statistics")
