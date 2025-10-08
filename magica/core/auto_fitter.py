"""
Automatic distribution fitting with model selection capabilities
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
import warnings

from .data_processor import DataProcessor
from .magic_adjuster import MagicAdjuster, get_available_distributions


class AutoFitter:
    """
    Automatic distribution fitting class with model selection capabilities.
    
    This class automatically tests multiple probability distributions and selects
    the best-fitting one based on specified criteria (default: RMSE).
    
    Uses lazy initialization pattern - MagicAdjuster instances are created only
    when needed for each distribution candidate.
    
    Parameters
    ----------
    data_processor : DataProcessor
        Processor instance with loaded data
    candidates : list of str, optional
        List of distribution names to test. If None, uses default set.
    criterion : str, default 'rmse'
        Selection criterion ('rmse', 'aic', 'bic', 'ks_pvalue', 'chi2_pvalue')
    
    Examples
    --------
    >>> import magica as ma
    >>> import numpy as np
    >>> 
    >>> # Load wind speed data
    >>> data = np.random.weibull(2, 1000) * 8 + 2
    >>> processor = ma.read_data(data)
    >>> 
    >>> # Auto-fit best distribution
    >>> auto_fitter = processor.get_auto_fitter()
    >>> best_result = auto_fitter.fit_best_distribution()
    >>> 
    >>> print(f"Best distribution: {best_result['distribution']}")
    >>> print(f"RMSE: {best_result['rmse']:.4f}")
    """
    
    def __init__(
        self, 
        data_processor: DataProcessor,
        candidates: Optional[List[str]] = None,
        criterion: str = 'rmse'
    ):
        """
        Initialize AutoFitter with data processor and configuration.
        
        Parameters
        ----------
        data_processor : DataProcessor
            The data processor containing the dataset to fit
        candidates : list of str, optional
            Distribution names to test. Default includes common distributions.
        criterion : str, default 'rmse'
            Selection criterion for best distribution
        """
        if data_processor.data is None:
            raise ValueError("DataProcessor must contain data before auto-fitting")
            
        self.data_processor = data_processor
        self.data = data_processor.get_data_array()
        
        # Get all available distributions from MagicAdjuster
        all_distributions = get_available_distributions()
        
        # Default candidate distributions (subset of stable, commonly used distributions)
        # User can override with candidates=list(get_available_distributions().keys()) for all
        if candidates is None:
            # Use a curated subset of the most stable distributions for environmental data
            stable_defaults = [
                'weibull_min', 'lognorm', 'gamma', 'norm', 'expon', 'rayleigh',
                'chi2', 'beta', 'uniform', 'logistic', 'gumbel_r', 'pareto',
                'invgamma', 'maxwell', 'triang', 'laplace'
            ]
            # Only keep distributions that exist in the full list
            self.candidates = [d for d in stable_defaults if d in all_distributions]
        else:
            # Validate user-provided candidates
            invalid_dists = [d for d in candidates if d not in all_distributions]
            if invalid_dists:
                raise ValueError(f"Invalid distributions: {invalid_dists}. "
                               f"Available: {sorted(all_distributions.keys())}")
            self.candidates = candidates
        
        self.criterion = criterion
        
        # Lazy initialization containers
        self._adjusters: Dict[str, Optional[MagicAdjuster]] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._best_distribution: Optional[str] = None
        self._comparison_complete = False
        
        # Initialize adjuster placeholders (not created yet!)
        for dist_name in self.candidates:
            self._adjusters[dist_name] = None
    
    def _get_adjuster(self, distribution: str) -> MagicAdjuster:
        """
        Factory method - creates MagicAdjuster for specific distribution when needed.
        
        This implements the lazy initialization pattern: adjusters are created
        only when first requested, not during __init__.
        
        Parameters
        ----------
        distribution : str
            Distribution name to get adjuster for
            
        Returns
        -------
        MagicAdjuster
            Adjuster instance (new or existing)
        """
        if distribution not in self.candidates:
            raise ValueError(f"Distribution '{distribution}' not in candidates: {self.candidates}")
            
        # Lazy creation - only create if doesn't exist yet
        if self._adjusters[distribution] is None:
            # Create new DataProcessor instance to avoid overwriting original
            temp_processor = DataProcessor()
            temp_processor.data = self.data_processor.data.copy()
            
            # Create and store the adjuster
            self._adjusters[distribution] = MagicAdjuster(temp_processor)
            
        return self._adjusters[distribution]
    

    
    def fit_single_distribution(self, distribution: str, **fit_kwargs) -> Dict[str, Any]:
        """
        Fit a single distribution and calculate all metrics.
        
        Parameters
        ----------
        distribution : str
            Distribution name to fit
        **fit_kwargs : dict
            Additional arguments passed to fit method
            
        Returns
        -------
        dict
            Comprehensive results for this distribution
        """
        try:
            # Get adjuster (created lazily if needed)
            adjuster = self._get_adjuster(distribution)
            
            # Fit the distribution
            adjuster.fit_distribution(distribution, **fit_kwargs)
            
            # Calculate all metrics using the goodness_of_fit method
            rmse = adjuster.goodness_of_fit('rmse')
            aic = adjuster.goodness_of_fit('aic')
            bic = adjuster.goodness_of_fit('bic')
            ks_result = adjuster.goodness_of_fit('ks')
            chi2_result = adjuster.goodness_of_fit('chi2', warn_on_normalization=False)
            
            # Compile results
            result = {
                'distribution': distribution,
                'parameters': adjuster.get_fitted_params(),
                'rmse': rmse,
                'aic': aic,
                'bic': bic,
                'ks_statistic': ks_result.get('ks_statistic', np.nan),
                'ks_pvalue': ks_result.get('p_value', np.nan),
                'chi2_statistic': chi2_result.get('chi2_statistic', np.nan),
                'chi2_pvalue': chi2_result.get('p_value', np.nan),
                'success': True,
                'adjuster': adjuster  # Store reference for later use
            }
            
            # Store in results cache
            self._results[distribution] = result
            return result
            
        except Exception as e:
            # Handle fitting failures gracefully
            error_result = {
                'distribution': distribution,
                'parameters': None,
                'rmse': float('inf'),
                'aic': float('inf'),
                'bic': float('inf'),
                'ks_statistic': np.nan,
                'ks_pvalue': np.nan,
                'chi2_statistic': np.nan,
                'chi2_pvalue': np.nan,
                'success': False,
                'error': str(e),
                'adjuster': None
            }
            
            self._results[distribution] = error_result
            warnings.warn(f"Failed to fit {distribution}: {e}")
            return error_result
    
    def fit_all_distributions(self, **fit_kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Fit all candidate distributions and return comprehensive results.
        
        Parameters
        ----------
        **fit_kwargs : dict
            Additional arguments passed to all fit methods
            
        Returns
        -------
        dict
            Results for all distributions, keyed by distribution name
        """
        print(f"Testing {len(self.candidates)} distributions...")
        
        for i, distribution in enumerate(self.candidates, 1):
            print(f"  [{i}/{len(self.candidates)}] Fitting {distribution}...")
            self.fit_single_distribution(distribution, **fit_kwargs)
            
        self._comparison_complete = True
        print("✓ All distributions fitted successfully")
        
        return self._results.copy()
    
    def fit_best_distribution(self, **fit_kwargs) -> Dict[str, Any]:
        """
        Automatically find and fit the best distribution based on criterion.
        
        Parameters
        ----------
        **fit_kwargs : dict
            Additional arguments passed to fit methods
            
        Returns
        -------
        dict
            Results for the best-fitting distribution
        """
        # Fit all distributions if not done yet
        if not self._comparison_complete:
            self.fit_all_distributions(**fit_kwargs)
        
        # Find best distribution based on criterion
        valid_results = {k: v for k, v in self._results.items() if v['success']}
        
        if not valid_results:
            raise RuntimeError("No distributions fitted successfully")
        
        # Selection logic based on criterion
        if self.criterion == 'rmse':
            best_dist = min(valid_results.keys(), key=lambda x: valid_results[x]['rmse'])
        elif self.criterion == 'aic':
            best_dist = min(valid_results.keys(), key=lambda x: valid_results[x]['aic'])
        elif self.criterion == 'bic':
            best_dist = min(valid_results.keys(), key=lambda x: valid_results[x]['bic'])
        elif self.criterion == 'ks_pvalue':
            best_dist = max(valid_results.keys(), key=lambda x: valid_results[x]['ks_pvalue'])
        elif self.criterion == 'chi2_pvalue':
            best_dist = max(valid_results.keys(), key=lambda x: valid_results[x]['chi2_pvalue'])
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
        
        self._best_distribution = best_dist
        return valid_results[best_dist]
    
    def get_comparison_table(self, sort_by: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get a formatted comparison table of all fitted distributions.
        
        Parameters
        ----------
        sort_by : str, optional
            Metric to sort by. If None, uses the selection criterion.
            
        Returns
        -------
        dict
            Sorted results table
        """
        if not self._comparison_complete:
            raise RuntimeError("Must call fit_all_distributions() first")
        
        sort_key = sort_by or self.criterion
        reverse = sort_key in ['ks_pvalue', 'chi2_pvalue']  # Higher is better for p-values
        
        # Sort results
        sorted_items = sorted(
            self._results.items(),
            key=lambda x: x[1].get(sort_key, float('inf')),
            reverse=reverse
        )
        
        return dict(sorted_items)
    
    def get_best_adjuster(self) -> MagicAdjuster:
        """
        Get the MagicAdjuster instance for the best-fitting distribution.
        
        Returns
        -------
        MagicAdjuster
            The adjuster fitted with the best distribution
        """
        if self._best_distribution is None:
            self.fit_best_distribution()
            
        return self._results[self._best_distribution]['adjuster']
    
    @staticmethod
    def get_all_available_distributions():
        """
        Get all distribution names available in MagicAdjuster.
        
        Returns
        -------
        list
            Sorted list of all available distribution names
        """
        return sorted(get_available_distributions().keys())
    
    def __repr__(self) -> str:
        """String representation of the AutoFitter."""
        status = "fitted" if self._comparison_complete else "not fitted"
        best = f", best={self._best_distribution}" if self._best_distribution else ""
        return f"AutoFitter(candidates={len(self.candidates)}, criterion={self.criterion}, {status}{best})"