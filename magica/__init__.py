"""
MagicA - Magic Adjustment

Python package for statistical data adjustment, with special focus on wind data.
Includes advanced fitting techniques, goodness-of-fit tests, and visualization.
"""

__version__ = "0.1.0"
__author__ = "Danilo Couto de Souza"
__email__ = "danilo.oceano@gmail.com"

# Main imports
from .core import DataProcessor

# Package metadata
__all__ = [
    "DataProcessor",
]
