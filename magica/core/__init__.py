"""
Core module - Main functionalities of MagicA
"""

from .data_processor import DataProcessor
from .magic_adjuster import MagicAdjuster, FitResult
from .auto_fitter import AutoFitter, EVA_FAMILIES
from .extremes_analyzer import ExtremesAnalyzer, EVAFit

__all__ = [
    "DataProcessor", "MagicAdjuster", "FitResult",
    "AutoFitter", "EVA_FAMILIES",
    "ExtremesAnalyzer", "EVAFit",
]
