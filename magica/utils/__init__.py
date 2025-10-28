"""
Utils module - Utility functions for data generation and analysis
"""

from .synthetic_data import generate_wind_data, generate_directional_wind_data

__all__ = [
    'generate_wind_data',
    'generate_directional_wind_data',
]
