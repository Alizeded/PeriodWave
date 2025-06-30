"""
Alias shim so `import periodwave` works.

所有符号都从 `model` 包 re-export。
"""
from importlib import import_module as _im
import sys as _sys

_mod = _im("model")
_sys.modules[__name__] = _mod
for _k, _v in _mod.__dict__.items():
    globals()[_k] = _v
del _im, _sys, _mod, _k, _v
