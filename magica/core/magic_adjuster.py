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
                # Complete and validated mapping of scipy.stats continuous distributions
                'alpha': stats.alpha,
                'anglit': stats.anglit,
                'arcsine': stats.arcsine,
                'argus': stats.argus,
                'beta': stats.beta,
                'betaprime': stats.betaprime,
                'bradford': stats.bradford,
                'burr': stats.burr,
                'burr12': stats.burr12,
                'cauchy': stats.cauchy,
                'chi': stats.chi,
                'chi2': stats.chi2,
                'cosine': stats.cosine,
                'crystalball': stats.crystalball,
                'dgamma': stats.dgamma,
                'dpareto_lognorm': stats.dpareto_lognorm,
                'dweibull': stats.dweibull,
                'erlang': stats.erlang,
                'expon': stats.expon,
                'exponnorm': stats.exponnorm,
                'exponpow': stats.exponpow,
                'exponweib': stats.exponweib,
                'f': stats.f,
                'fatiguelife': stats.fatiguelife,
                'fisk': stats.fisk,
                'foldcauchy': stats.foldcauchy,
                'foldnorm': stats.foldnorm,
                'gamma': stats.gamma,
                'gausshyper': stats.gausshyper,
                'genexpon': stats.genexpon,
                'genextreme': stats.genextreme,
                'gengamma': stats.gengamma,
                'genhalflogistic': stats.genhalflogistic,
                'genhyperbolic': stats.genhyperbolic,
                'geninvgauss': stats.geninvgauss,
                'genlogistic': stats.genlogistic,
                'gennorm': stats.gennorm,
                'genpareto': stats.genpareto,
                'gibrat': stats.gibrat,
                'gompertz': stats.gompertz,
                'gumbel_l': stats.gumbel_l,
                'gumbel_r': stats.gumbel_r,
                'halfcauchy': stats.halfcauchy,
                'halfgennorm': stats.halfgennorm,
                'halflogistic': stats.halflogistic,
                'halfnorm': stats.halfnorm,
                'hypsecant': stats.hypsecant,
                'invgamma': stats.invgamma,
                'invgauss': stats.invgauss,
                'invweibull': stats.invweibull,
                'irwinhall': stats.irwinhall,
                'jf_skew_t': stats.jf_skew_t,
                'johnsonsb': stats.johnsonsb,
                'johnsonsu': stats.johnsonsu,
                'kappa3': stats.kappa3,
                'kappa4': stats.kappa4,
                'ksone': stats.ksone,
                'kstwo': stats.kstwo,
                'kstwobign': stats.kstwobign,
                'landau': stats.landau,
                'laplace': stats.laplace,
                'laplace_asymmetric': stats.laplace_asymmetric,
                'levy': stats.levy,
                'levy_l': stats.levy_l,
                'levy_stable': stats.levy_stable,
                'loggamma': stats.loggamma,
                'logistic': stats.logistic,
                'loglaplace': stats.loglaplace,
                'lognorm': stats.lognorm,
                'loguniform': stats.loguniform,
                'lomax': stats.lomax,
                'maxwell': stats.maxwell,
                'mielke': stats.mielke,
                'moyal': stats.moyal,
                'nakagami': stats.nakagami,
                'ncf': stats.ncf,
                'nct': stats.nct,
                'ncx2': stats.ncx2,
                'norm': stats.norm,
                'norminvgauss': stats.norminvgauss,
                'pareto': stats.pareto,
                'pearson3': stats.pearson3,
                'powerlaw': stats.powerlaw,
                'powerlognorm': stats.powerlognorm,
                'powernorm': stats.powernorm,
                'rayleigh': stats.rayleigh,
                'rdist': stats.rdist,
                'recipinvgauss': stats.recipinvgauss,
                'reciprocal': stats.reciprocal,
                'rel_breitwigner': stats.rel_breitwigner,
                'rice': stats.rice,
                'semicircular': stats.semicircular,
                'skewcauchy': stats.skewcauchy,
                'skewnorm': stats.skewnorm,
                'studentized_range': stats.studentized_range,
                'students_t': stats.t,  # More descriptive name for t-distribution
                't': stats.t,  # Also support direct 't' name
                'trapezoid': stats.trapezoid,
                'trapz': stats.trapezoid,  # Common alias
                'triang': stats.triang,
                'truncexpon': stats.truncexpon,
                'truncnorm': stats.truncnorm,
                'truncpareto': stats.truncpareto,
                'truncweibull_min': stats.truncweibull_min,
                'tukeylambda': stats.tukeylambda,
                'uniform': stats.uniform,
                'vonmises': stats.vonmises,
                'vonmises_line': stats.vonmises_line,
                'wald': stats.wald,
                'weibull': stats.weibull_min,  # Common alias for Weibull
                'weibull_max': stats.weibull_max,
                'weibull_min': stats.weibull_min,
                'wrapcauchy': stats.wrapcauchy
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
    
    def __getattr__(self, name):
        """
        Delegate method calls to the fitted scipy distribution with smart defaults.
        
        This allows direct access to all scipy.stats methods like:
        cdf, pdf, ppf, sf, isf, rvs, stats, etc.
        
        For methods that commonly evaluate distributions at data points 
        (pdf, cdf, sf, logpdf, logcdf, logsf), if no input is provided,
        the original data will be used automatically.
        """
        if self.fitted_distribution is None or self.fitted_params is None:
            raise ValueError("No distribution has been fitted yet. Call fit_distribution() first.")
        
        # Get the frozen distribution
        frozen_dist = self.fitted_distribution(*self.fitted_params)
        
        # Check if the method exists in the distribution
        if not hasattr(frozen_dist, name):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Methods that commonly use the original data as default input
        data_aware_methods = {
            'pdf', 'cdf', 'sf', 'logpdf', 'logcdf', 'logsf', 
            'ppf', 'isf', 'logpdf', 'interval'
        }
        
        original_method = getattr(frozen_dist, name)
        
        if name in data_aware_methods:
            def smart_wrapper(*args, **kwargs):
                """
                Smart wrapper that uses original data as default for common methods.
                
                If the first argument is not provided for evaluation methods,
                use the original data points.
                """
                # If no positional arguments provided, use original data
                if len(args) == 0 and name in ['pdf', 'cdf', 'sf', 'logpdf', 'logcdf', 'logsf']:
                    return original_method(self.data, **kwargs)
                else:
                    # Normal call with provided arguments
                    return original_method(*args, **kwargs)
            
            # Copy metadata from original method
            smart_wrapper.__name__ = getattr(original_method, '__name__', name)
            smart_wrapper.__doc__ = getattr(original_method, '__doc__', None)
            
            return smart_wrapper
        else:
            # For other methods, delegate normally
            return original_method
        
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
    
    def get_bin_number_sturges(self):
        """
        Calculate the optimal number of bins using Sturges' rule.

        Sturges' rule is a simple heuristic for determining the number of bins in a histogram.
        It assumes that the data follows a normal distribution and is best suited for smaller datasets.
        """
        N = len(self.data)
        return int(1 + np.log2(N))
    
    def get_bin_number_rice(self):
        """
        Calculate the optimal number of bins using Rice's rule.

        Rice's rule suggests a bin count that scales with the cube root of the dataset size.
        It is a simple alternative to Sturges' rule and works well for larger datasets.
        """
        N = len(self.data)
        return int(2 * N**(1/3))

    def get_bin_number_freedman_diaconis(self):
        """
        Calculate the number of bins based on the Freedman-Diaconis rule.
        This rule uses the interquartile range (IQR) to calculate bin width and is robust
        for skewed distributions.
        """
        iqr = np.percentile(self.data, 75) - np.percentile(self.data, 25)
        bin_width = 2 * iqr / len(self.data)**(1/3)
        return max(1, int((max(self.data) - min(self.data)) / bin_width))

    def get_bin_number_scott(self):
        """
        Calculate the bin width based on Scott's rule.
        Scott's rule minimizes the integrated mean squared error for normal distributions,
        but can also work for large datasets.
        """
        bin_width = 3.5 * np.std(self.data) / len(self.data)**(1/3)
        return max(1, int((max(self.data) - min(self.data)) / bin_width))
    
    def get_bin_number_doane(self):
        """
        Calculate the optimal number of bins using Doane's formula.

        Doane's formula is an extension of Sturges' rule that accounts for the skewness 
        of the data distribution. This method is particularly useful when dealing with 
        non-normal data distributions, as it adjusts the bin count based on the sample skewness.
        """
        N = len(self.data)
        g1 = stats.skew(self.data)
        sigma_g1 = np.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
        return int(1 + np.log2(N) + np.log2(1 + abs(g1) / sigma_g1))

    def get_num_bins(self, bins='doane'):
        """
        Determines the number of bins for histogram plotting based on a chosen method.

        Args:
            bins (int or str, optional): 
                The binning method to use. Default is `"doane"`.
                Options:
                    - Integer (e.g., `30`): A fixed number of bins.
                    - 'sturges': Sturges' rule (log-based, best for normal distributions).
                    - 'freedman-diaconis': Uses Freedman-Diaconis rule (best for skewed distributions).
                    - 'rice': Uses Rice’s rule (scales with cube root of dataset size).
                    - 'scott': Uses Scott’s rule (minimizes IMSE for normal distributions).
                    - 'doane': Uses Doane's rule (default, extension of Sturges' rule that accounts for the skewness of the data distribution).

        Returns:
            int: The computed number of bins.

        Raises:
            ValueError: If an unsupported binning method is provided.

        Example:
            >>> lm_adjust = PointAdjustment(data, bins='freedman-diaconis')
            >>> num_bins = lm_adjust.get_num_bins('freedman-diaconis')
            >>> print(num_bins)
            25
        """
        N = len(self.data)
        if bins == 'sturges':
            num_bins = self.get_bin_number_sturges()
        elif bins == 'rice':
            return self.get_bin_number_rice()
        elif bins == 'scott':
            num_bins = self.get_bin_number_scott()
        elif bins == 'freedman-diaconis':
            num_bins = self.get_bin_number_freedman_diaconis()
        elif bins == 'doane':
            num_bins = self.get_bin_number_doane()
        else:
            num_bins = bins  # If a specific number of bins is provided

        return num_bins

    def goodness_of_fit(self, method: str, bins: Union['str', 'int'] = 'doane'):
        if method in ['chisquared', 'chi2']:
            # Determine number of bins
            n_bins = self.get_num_bins(bins)

            dist_info = self.get_distribution_info()
            
            params = self.fitted_params

            # Empirical frequencies
            observed_freq, bin_edges = np.histogram(self.data, bins=n_bins)

            # Compute theoretical frequencies
            expected_freq_weibull = len(self.data) * np.diff(self.fitted_distribution.cdf(bin_edges, *params))
            
            # Normalize the expected frequencies to match the total number of data points
            expected_freq_weibull *= observed_freq.sum() / expected_freq_weibull.sum()  # Normalizando para corresponder à soma das observadas

            return stats.chisquare(observed_freq, f_exp=expected_freq_weibull)
        
## TO-DO:
# When set a new distribution to an already used variable, it will override the distribution form previous variable
# For example:
# fitted_data_weibull = data.fit_distribution('weibull')
# fitted_data_weibull = data.fit_distribution('weibull')
#
# The distribution from "fitted_data_weibull" will return the norm dist.

 