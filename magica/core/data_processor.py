"""
Data processor for statistical analysis
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
import warnings


class DataProcessor:
    """
    Simple class for loading and basic processing of data.
    
    This class provides basic methods to load and validate data
    for statistical analysis, with a focus on wind data.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.data = None
        self.metadata = {}
        
    def load_data(self, 
                  data: Union[str, pd.DataFrame, np.ndarray, list], 
                  **kwargs) -> 'DataProcessor':
        """
        Load data from different sources.
        
        Parameters
        ----------
        data : str, DataFrame, ndarray, or list
            Data to be loaded. Can be:
            - String: path to CSV/Excel file
            - DataFrame: pandas data
            - ndarray: numpy array
            - List: list of values
        **kwargs : dict
            Additional arguments for pd.read_csv() or pd.read_excel()
            
        Returns
        -------
        DataProcessor
            Processor instance with loaded data
        """
        if isinstance(data, str):
            # Load from file
            if data.endswith('.csv'):
                self.data = pd.read_csv(data, **kwargs)
            elif data.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(data, **kwargs)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
                
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
            
        elif isinstance(data, (np.ndarray, list)):
            self.data = pd.Series(data, name='data')
            
        else:
            raise TypeError("Unsupported data type.")
            
        self._update_metadata()
        return self
    
    def get_data_array(self) -> np.ndarray:
        """
        Return data as 1D numpy array.
        
        Returns
        -------
        ndarray
            1D numpy array with the data
        """
        if self.data is None:
            raise ValueError("No data has been loaded.")
            
        if isinstance(self.data, pd.DataFrame):
            return self.data.values.flatten()
        else:
            return self.data.values
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Return basic descriptive statistics of the data.
        
        Returns
        -------
        dict
            Dictionary with basic statistics
        """
        if self.data is None:
            raise ValueError("No data has been loaded.")
            
        data_array = self.get_data_array()
        
        return {
            'count': len(data_array),
            'mean': np.mean(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'median': np.median(data_array)
        }
    
    def _update_metadata(self):
        """Update data metadata."""
        if self.data is not None:
            self.metadata['shape'] = self.data.shape
            self.metadata['type'] = type(self.data).__name__
            if isinstance(self.data, pd.DataFrame):
                self.metadata['columns'] = list(self.data.columns)
            self.metadata['last_updated'] = pd.Timestamp.now()
    
    def __repr__(self) -> str:
        """String representation of the object."""
        if self.data is None:
            return "DataProcessor(no data loaded)"
        return f"DataProcessor(shape={self.metadata.get('shape', 'unknown')})"
