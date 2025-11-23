"""Encoders package: use lazy imports to avoid heavy side-effects on import."""

from typing import Any
import importlib

__all__ = ['FastEventPatternEncoder', 'DualLayerSRFFN']


def __getattr__(name: str) -> Any:
    if name == 'FastEventPatternEncoder':
        mod = importlib.import_module('.fast_event_encoder', __name__)
        return getattr(mod, 'FastEventPatternEncoder')
    if name == 'DualLayerSRFFN':
        mod = importlib.import_module('.dual_layer_srffn', __name__)
        return getattr(mod, 'DualLayerSRFFN')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")