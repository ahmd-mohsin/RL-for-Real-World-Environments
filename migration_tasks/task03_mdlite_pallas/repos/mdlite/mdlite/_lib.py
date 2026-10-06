"""ctypes binding of libmdlite.so (built from ``src/`` by setup.py)."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

_c_double_p = ctypes.POINTER(ctypes.c_double)


def _load():
    here = Path(__file__).resolve().parent
    cands = [os.environ.get("MDLITE_LIBRARY", ""), str(here / "libmdlite.so")]
    for c in cands:
        if c and Path(c).exists():
            lib = ctypes.CDLL(c)
            break
    else:
        raise ImportError("libmdlite.so not found; build mdlite with `pip install .`")
    vp, cp, i, ll = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_longlong
    sig = {
        "mdl_last_error": (cp, []),
        "mdl_term_count": (i, []),
        "mdl_term_name": (cp, [i]),
        "mdl_create": (vp, [cp, cp, cp]),
        "mdl_destroy": (None, [vp]),
        "mdl_natoms": (i, [vp]),
        "mdl_step": (ll, [vp]),
        "mdl_dof": (i, [vp]),
        "mdl_run": (i, [vp, ll]),
        "mdl_finish": (i, [vp]),
        "mdl_energies": (i, [vp, _c_double_p, _c_double_p, _c_double_p]),
        "mdl_forces": (i, [vp, _c_double_p]),
        "mdl_get_positions": (i, [vp, _c_double_p]),
        "mdl_get_velocities": (i, [vp, _c_double_p]),
        "mdl_set_positions": (i, [vp, _c_double_p]),
        "mdl_set_velocities": (i, [vp, _c_double_p]),
        "mdl_get_box": (i, [vp, _c_double_p]),
        "mdl_set_box": (i, [vp, _c_double_p]),
        "mdl_save_checkpoint": (i, [vp, cp]),
        "mdl_load_checkpoint": (i, [vp, cp]),
        "mdl_info": (i, [vp, ctypes.c_char_p, i]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args
    return lib


lib = _load()


def ptr(a):
    return a.ctypes.data_as(_c_double_p)
