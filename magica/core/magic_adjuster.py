"""
Statistical distribution fitting and adjustment for wind data
"""

import numpy as np
from scipy import stats
from typing import Union, Dict, Any, Optional, Tuple
import warnings

from .data_processor import DataProcessor


class MagicAdjuster:
    """
    Simple class for statistical distribution fitting using SciPy.
    
    This class takes processed data and fits statistical distributions,
    designed to be extended with goodness-of-fit tests and advanced features.
    """
    
    def __init__(self, data_processor: DataProcessor):
        """
        Initialize the adjuster with processed data.
        
        Parameters
        ----------
        data_processor : DataProcessor
            Processor instance with loaded data
        """
        if data_processor.data is None:
            raise ValueError("DataProcessor must have data loaded.")
        
        self.data_processor = data_processor
        self.data = data_processor.get_data_array()
        self.fitted_distribution = None
        self.fitted_params = None
        self.distribution_name = None
        
    def fit_distribution(self, distribution: Union[str, object], **kwargs) -> 'MagicAdjuster':
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
        MagicAdjuster
            Adjuster instance with fitted distribution
        """
        # Handle string distribution names
        if isinstance(distribution, str):
            distribution_map = {
                'weibull': stats.weibull_min,
                'gamma': stats.gamma,
                'lognorm': stats.lognorm,
                'norm': stats.norm,
                'exponential': stats.expon,
                'beta': stats.beta,
                'uniform': stats.uniform,
                'pareto': stats.pareto
            }
            
            if distribution.lower() not in distribution_map:
                available = list(distribution_map.keys())
                raise ValueError(f"Unknown distribution '{distribution}'. Available: {available}")
            
            self.fitted_distribution = distribution_map[distribution.lower()]
            self.distribution_name = distribution.lower()
        else:
            # Assume it's a SciPy distribution object
            self.fitted_distribution = distribution
            self.distribution_name = getattr(distribution, 'name', str(distribution))
        
        # Fit the distribution
        try:
            self.fitted_params = self.fitted_distribution.fit(self.data, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to fit {self.distribution_name} distribution: {e}")
        
        return self
    
    def get_fitted_params(self) -> Tuple:
        """
        Get the fitted distribution parameters.
        
        Returns
        -------
        tuple
            Fitted parameters of the distribution
        """
        if self.fitted_params is None:
            raise ValueError("No distribution has been fitted yet.")
        
        return self.fitted_params
    
    def get_distribution_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted distribution.
        
        Returns
        -------
        dict
            Dictionary with distribution information
        """
        if self.fitted_distribution is None:
            raise ValueError("No distribution has been fitted yet.")
        
        return {
            'name': self.distribution_name,
            'parameters': self.fitted_params,
            'num_params': len(self.fitted_params),
            'data_size': len(self.data)
        }
    
    def __repr__(self) -> str:
        """String representation of the object."""
        if self.fitted_distribution is None:
            return f"MagicAdjuster(data_size={len(self.data)}, no distribution fitted)"
        return f"MagicAdjuster(data_size={len(self.data)}, distribution='{self.distribution_name}')"
