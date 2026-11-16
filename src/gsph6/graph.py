"""Backward-compatibility shim: aliases the canonical cspph implementation so legacy `runners.*`/`gsph6.*` import paths keep working (including private names). New code should use `cspph.*`.
"""
import sys as _sys
import importlib as _il

_m = _il.import_module('cspph.graph.patch_digraph')
_sys.modules[__name__] = _m
