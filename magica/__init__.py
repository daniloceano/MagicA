"""
MagicA - Magic Adjustment

Python package for statistical data adjustment, with special focus on wind data.
Includes advanced fitting techniques, goodness-of-fit tests, and visualization.
"""

import numpy as np
import pandas as pd
from typing import Union

__version__ = "0.2.0"
__author__ = "Danilo Couto de Souza"
__email__ = "danilo.oceano@gmail.com"

# Main imports
from .core import DataProcessor, FitResult


def read_data(data: Union[np.ndarray, list, pd.Series, pd.DataFrame]) -> DataProcessor:
    """
    Read and process data for statistical analysis.
    
    This is the main entry point for loading data into MagicA.
    Equivalent to pandas.read_csv() but for array-like data.
    
    Parameters
    ----------
    data : array-like
        Data to be processed. Can be:
        - numpy array
        - list of values
        - pandas Series
        - pandas DataFrame (will be flattened)
        
    Returns
    -------
    DataProcessor
        Processor instance with loaded data, ready for analysis
        
    Examples
    --------
    >>> import magica as ma, numpy as np
    >>> wind_data = np.random.weibull(2, 500) * 8
    >>> data = ma.read_data(wind_data)
    >>> fit = data.fit('weibull', floc=0)
    >>> fit.params
    >>> fit.cdf(np.array([5., 10.]))
    >>> fit.goodness_of_fit('ks')
    """
    return DataProcessor(data)


# Package metadata
__all__ = [
    "DataProcessor",
    "FitResult",
    "read_data",
]
