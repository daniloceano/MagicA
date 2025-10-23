"""
Core module - Main functionalities of MagicA
"""

from .data_processor import DataProcessor
from .magic_adjuster import MagicAdjuster
from .auto_fitter import AutoFitter
from .extremes_analyzer import ExtremesAnalyzer

__all__ = ["DataProcessor", "MagicAdjuster", "AutoFitter", "ExtremesAnalyzer"]
