import juliapkg  # noqa: I001, F401
from pathlib import Path

from . import _sysimage

_sysimage.configure()

# To change the version of Finch used, see the documentation for pyjuliapkg here: https://github.com/JuliaPy/pyjuliapkg
# Use pyjuliapkg to modify the `juliapkg.json` file in the root of this repo.
# You can also run `develop.py` to quickly use a local copy of Finch.jl.
# An example development json is found in `juliapkg_dev.json`
# Julia and all packages are pinned so the prebuilt sysimage matches; update the
# pins with `scripts/sysimage/pin_julia_deps.py`.
import juliacall as jc  # noqa: E402, F401
from juliacall import Main as jl  # noqa: E402, F401

jl.seval("using Finch")
jl.Base.include(jl.Finch.Galley, str(Path(__file__).with_name("finch_patches.jl")))
jl.seval("using HDF5")
jl.seval("using NPZ")
jl.seval("using TensorMarket")
jl.seval("using Random")
jl.seval("using Statistics")
