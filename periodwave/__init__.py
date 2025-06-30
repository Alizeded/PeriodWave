"""Alias shim so `import periodwave` works and exposes root-level helpers."""

from importlib import import_module as _im
import sys as _sys

_mod = _im("model")
_sys.modules[__name__] = _mod
for _k, _v in _mod.__dict__.items():
    globals()[_k] = _v

_extra = ("encodec_feature_extractor", "extract_energy", "meldataset_prior_length")
for _name in _extra:
    _m = _im(_name)
    setattr(_sys.modules[__name__], _name, _m)
    globals()[_name] = _m

del _im, _sys, _mod, _k, _v, _extra, _m, _name
