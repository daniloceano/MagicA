"""
Data processor for statistical analysis
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple
import warnings


class DataProcessor:
    """
    Simple class for loading and basic processing of numerical data.
    
    This class provides basic methods to load and validate numerical data
    for statistical analysis, converting data to numpy arrays.
    """
    
    def __init__(self, data: Union[np.ndarray, list, pd.Series, pd.DataFrame] = None):
        """
        Initialize the data processor.
        
        Parameters
        ----------
        data : array-like, optional
            Data to be processed. Can be:
            - numpy array
            - list of values  
            - pandas Series
            - pandas DataFrame (will be flattened)
        """
        self.data = None
        self.metadata = {}
        
        # Internal adjuster for distribution fitting
        self._adjuster = None
        
        if data is not None:
            self.load_data(data)
        
    def load_data(self, data: Union[np.ndarray, list, pd.Series, pd.DataFrame]) -> 'DataProcessor':
        """
        Load data and convert to numpy array.
        
        Parameters
        ----------
        data : array-like
            Data to be loaded. Can be:
            - numpy array
            - list of values
            - pandas Series
            - pandas DataFrame (will be flattened)
            
        Returns
        -------
        DataProcessor
            Processor instance with loaded data
        """
        if isinstance(data, np.ndarray):
            self.data = data.flatten() if data.ndim > 1 else data.copy()
            
        elif isinstance(data, list):
            self.data = np.array(data, dtype=float)
            
        elif isinstance(data, pd.Series):
            self.data = data.values
            
        elif isinstance(data, pd.DataFrame):
            self.data = data.values.flatten()
            
        else:
            # Try to convert any array-like object
            try:
                self.data = np.array(data, dtype=float).flatten()
            except (ValueError, TypeError):
                raise TypeError(f"Cannot convert data of type {type(data)} to numpy array.")
        
        # Remove NaN values and warn if any were found
        if np.any(np.isnan(self.data)):
            n_nan = np.sum(np.isnan(self.data))
            warnings.warn(f"Found {n_nan} NaN values. They will be removed.")
            self.data = self.data[~np.isnan(self.data)]
        
        if len(self.data) == 0:
            raise ValueError("No valid data points after removing NaN values.")
            
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
            
        return self.data.copy()
    
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
        
        return {
            'count': len(self.data),
            'mean': np.mean(self.data),
            'std': np.std(self.data, ddof=1),  # Sample standard deviation
            'var': np.var(self.data, ddof=1),  # Sample variance
            'min': np.min(self.data),
            'max': np.max(self.data),
            'median': np.median(self.data),
            'q25': np.percentile(self.data, 25),
            'q75': np.percentile(self.data, 75)
        }
    
    def _get_adjuster(self):
        """Get or create the internal adjuster."""
        if self._adjuster is None:
            from .magic_adjuster import MagicAdjuster
            self._adjuster = MagicAdjuster(self)
        return self._adjuster
    
    def fit_distribution(self, distribution: Union[str, object], **kwargs) -> 'DataProcessor':
        """
        Fit a statistical distribution to the data.
        
        Parameters
        ----------
        distribution : str or scipy.stats distribution
            Distribution to fit. Can be:
            - String: 'weibull', 'gamma', 'lognorm', 'norm', etc.
            - SciPy distribution object: stats.weibull_min, stats.gamma, etc.
        **kwargs : dict
            Additional arguments passed to the distribution's fit method
            
        Returns
        -------
        DataProcessor
            Processor instance with fitted distribution
        """
        adjuster = self._get_adjuster()
        adjuster.fit_distribution(distribution, **kwargs)
        return self
    
    def get_fitted_params(self) -> Tuple:
        """
        Get the fitted distribution parameters.
        
        Returns
        -------
        tuple
            Fitted parameters of the distribution
        """
        if self._adjuster is None:
            raise ValueError("No distribution has been fitted yet.")
        
        return self._adjuster.get_fitted_params()
    
    def get_distribution_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted distribution.
        
        Returns
        -------
        dict
            Dictionary with distribution information
        """
        if self._adjuster is None:
            raise ValueError("No distribution has been fitted yet.")
        
        return self._adjuster.get_distribution_info()
    
    def get_auto_fitter(self, candidates=None, criterion='rmse'):
        """
        Create an AutoFitter instance for automatic distribution selection.
        
        This method follows the factory pattern - it creates an AutoFitter
        instance when needed, allowing automatic testing of multiple distributions
        to find the best fit based on specified criteria.
        
        Parameters
        ----------
        candidates : list of str, optional
            List of distribution names to test. If None, uses default set including:
            'weibull_min', 'lognorm', 'gamma', 'norm', 'expon', 'rayleigh', 'chi2', 'beta'
        criterion : str, default 'rmse'
            Selection criterion ('rmse', 'aic', 'bic', 'ks_pvalue', 'chi2_pvalue')
            
        Returns
        -------
        AutoFitter
            AutoFitter instance configured with this data
            
        Examples
        --------
        >>> import magica as ma
        >>> import numpy as np
        >>> 
        >>> # Load data
        >>> data = np.random.weibull(2, 1000)
        >>> processor = ma.read_data(data)
        >>> 
        >>> # Get auto fitter and find best distribution
        >>> auto_fitter = processor.get_auto_fitter()
        >>> best_result = auto_fitter.fit_best_distribution()
        >>> print(f"Best distribution: {best_result['distribution']}")
        """
        # Import here to avoid circular imports
        from .auto_fitter import AutoFitter
        
        if self.data is None:
            raise ValueError("No data has been loaded.")
            
        return AutoFitter(self, candidates=candidates, criterion=criterion)
    
    def _update_metadata(self):
        """Update data metadata."""
        if self.data is not None:
            self.metadata['length'] = len(self.data)
            self.metadata['dtype'] = str(self.data.dtype)
            self.metadata['last_updated'] = pd.Timestamp.now()
    
    def __repr__(self) -> str:
        """String representation of the object."""
        if self.data is None:
            return "DataProcessor(no data loaded)"
        
        # Check if distribution is fitted via adjuster
        dist_info = ""
        if self._adjuster is not None and self._adjuster.distribution_name is not None:
            dist_info = f", distribution='{self._adjuster.distribution_name}'"
            
        return f"DataProcessor(length={len(self.data)}, dtype={self.data.dtype}{dist_info})"
    
    def __len__(self) -> int:
        """Return the length of the data."""
        if self.data is None:
            return 0
        return len(self.data)
    
    def __getattr__(self, name):
        """
        Delegate method calls to the internal MagicAdjuster.
        
        This allows direct access to all scipy.stats methods like:
        cdf, pdf, ppf, sf, isf, rvs, stats, etc.
        
        Parameters
        ----------
        name : str
            Name of the method/attribute to access
            
        Returns
        -------
        Any
            Method or attribute from the fitted distribution
        """
        # First check if we have an adjuster with a fitted distribution
        if self._adjuster is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                               "Did you forget to call fit_distribution() first?")
        
        # Delegate to the adjuster's __getattr__
        try:
            return getattr(self._adjuster, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
