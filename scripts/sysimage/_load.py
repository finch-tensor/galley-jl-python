"""Load ``galley_jl_python._sysimage`` by path.

Importing it through the package would run ``galley_jl_python/__init__.py``,
which starts Julia.
"""

import importlib.util
from pathlib import Path

_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "galley_jl_python" / "_sysimage.py"
)
_spec = importlib.util.spec_from_file_location("_galley_sysimage", _PATH)
sysimage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sysimage)
