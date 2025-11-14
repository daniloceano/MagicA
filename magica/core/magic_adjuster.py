"""
Statistical distribution fitting and adjustment for wind data
"""

from matplotlib.pylab import seed
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import xarray as xr
from typing import Union, Dict, Any, Optional, Tuple, List
import warnings
from tqdm import tqdm, trange

from .data_processor import DataProcessor


def get_available_distributions():
    """
    Get all available distribution names and their scipy.stats objects.
    
    Returns
    -------
    dict
        Dictionary mapping distribution names to scipy.stats objects
    """
    return {
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
        #'levy_stable': stats.levy_stable,
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
        #'studentized_range': stats.studentized_range,
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
            distribution_map = get_available_distributions()
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
        
        # Check if the method exists in the distribution class (not frozen)
        if not hasattr(self.fitted_distribution, name):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Methods that commonly use the original data as default input
        data_aware_methods = {
            'pdf', 'cdf', 'sf', 'logpdf', 'logcdf', 'logsf', 
            'ppf', 'isf', 'interval'
        }
        
        # Get the original method from the distribution class
        original_method = getattr(self.fitted_distribution, name)
        
        if name in data_aware_methods:
            def smart_wrapper(*args, **kwargs):
                """
                Smart wrapper that uses original data as default for common methods.
                
                If the first argument is not provided for evaluation methods,
                use the original data points. If parameters are not provided,
                use the fitted parameters.
                """
                # For methods that evaluate at data points, use smart defaults
                if len(args) == 0 and name in ['pdf', 'cdf', 'sf', 'logpdf', 'logcdf', 'logsf']:
                    return original_method(self.data, *self.fitted_params, **kwargs)
                # If only data provided (1 arg), add fitted parameters
                elif len(args) == 1 and name in ['pdf', 'cdf', 'sf', 'logpdf', 'logcdf', 'logsf']:
                    return original_method(args[0], *self.fitted_params, **kwargs)
                # For ppf, isf, interval - these need explicit input, use frozen distribution
                elif name in ['ppf', 'isf', 'interval']:
                    frozen_dist = self.fitted_distribution(*self.fitted_params)
                    return getattr(frozen_dist, name)(*args, **kwargs)
                else:
                    # Normal call with all provided arguments
                    return original_method(*args, **kwargs)
            
            # Copy metadata from original method
            smart_wrapper.__name__ = getattr(original_method, '__name__', name)
            smart_wrapper.__doc__ = getattr(original_method, '__doc__', None)
            
            return smart_wrapper
        else:
            # For other methods, use frozen distribution for convenience
            frozen_dist = self.fitted_distribution(*self.fitted_params)
            return getattr(frozen_dist, name)
        
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
        
        Returns
        -------
        int
            Number of bins calculated using Sturges' rule
        """
        N = len(self.data)
        return int(1 + np.log2(N))
    
    def get_bin_number_rice(self):
        """
        Calculate the optimal number of bins using Rice's rule.
        
        Rice's rule suggests a bin count that scales with the cube root of the dataset size.
        It is a simple alternative to Sturges' rule and works well for larger datasets.
        
        Returns
        -------
        int
            Number of bins calculated using Rice's rule
        """
        N = len(self.data)
        return int(2 * N**(1/3))

    def get_bin_number_freedman_diaconis(self):
        """
        Calculate the number of bins based on the Freedman-Diaconis rule.
        
        This rule uses the interquartile range (IQR) to calculate bin width and is robust
        for skewed distributions.
        
        Returns
        -------
        int
            Number of bins calculated using Freedman-Diaconis rule
        """
        iqr = np.percentile(self.data, 75) - np.percentile(self.data, 25)
        bin_width = 2 * iqr / len(self.data)**(1/3)
        return max(1, int((max(self.data) - min(self.data)) / bin_width))

    def get_bin_number_scott(self):
        """
        Calculate the bin width based on Scott's rule.
        
        Scott's rule minimizes the integrated mean squared error for normal distributions,
        but can also work for large datasets.
        
        Returns
        -------
        int
            Number of bins calculated using Scott's rule
        """
        bin_width = 3.5 * np.std(self.data) / len(self.data)**(1/3)
        return max(1, int((max(self.data) - min(self.data)) / bin_width))
    
    def get_bin_number_doane(self):
        """
        Calculate the optimal number of bins using Doane's formula.

        Doane's formula is an extension of Sturges' rule that accounts for the skewness 
        of the data distribution. This method is particularly useful when dealing with 
        non-normal data distributions, as it adjusts the bin count based on the sample skewness.
        
        Returns
        -------
        int
            Number of bins calculated using Doane's rule
        """
        N = len(self.data)
        g1 = stats.skew(self.data)
        sigma_g1 = np.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
        return int(1 + np.log2(N) + np.log2(1 + abs(g1) / sigma_g1))

    def get_num_bins(self, bins='doane'):
        """
        Determines the number of bins for histogram plotting based on a chosen method.

        Parameters
        ----------
        bins : int or str, optional 
            The binning method to use. Default is 'doane'.
            
            - Integer (e.g., 30): A fixed number of bins.
            - 'sturges': Sturges' rule (log-based, best for normal distributions).
            - 'freedman-diaconis': Uses Freedman-Diaconis rule (best for skewed distributions).
            - 'rice': Uses Rice’s rule (scales with cube root of dataset size).
            - 'scott': Uses Scott’s rule (minimizes IMSE for normal distributions).
            - 'doane': Uses Doane's rule (extension of Sturges' rule that accounts for the skewness of the data distribution).

        Returns
        -------
        int
            The computed number of bins.

        Raises
        ------
        ValueError
            If an unsupported binning method is provided.
        """
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

    def goodness_of_fit(self, method: str, bins: Union[str, int] = 'doane', warn_on_normalization: bool = True):
        """
        Perform goodness-of-fit test on the fitted distribution.
        
        Parameters
        ----------
        method : str
            Test method: 'chi2', 'ks', 'rmse', 'aic', or 'bic'
        bins : int or str, default='doane'
            Binning method for chi2 and rmse tests
        warn_on_normalization : bool, default=True
            Whether to warn about frequency normalization in chi2 test
            
        Returns
        -------
        dict or float
            Test results (dict for statistical tests, float for RMSE/AIC/BIC)
        """
        if self.fitted_distribution is None or self.fitted_params is None:
            raise ValueError("No distribution has been fitted yet. Call fit_distribution() first.")
        
        if method.lower() in ['chisquared', 'chi2']:
            n_bins = self.get_num_bins(bins)
            params = self.fitted_params

            # Empirical frequencies  
            observed_freq, bin_edges = np.histogram(self.data, bins=n_bins)
            
            # Compute theoretical probabilities for each bin
            bin_probs = np.diff(self.fitted_distribution.cdf(bin_edges, *params))
            
            # Expected frequencies = probabilities × total sample size
            expected_freq = bin_probs * len(self.data)

            # Check if normalization is needed
            discrepancy = abs(expected_freq.sum() - observed_freq.sum())
            if discrepancy > 1e-6 and warn_on_normalization:
                warnings.warn(f"Normalizing expected frequencies. Original sum: {expected_freq.sum():.6f}, "
                            f"Target sum: {observed_freq.sum()}")

            # Normalize only if necessary
            if discrepancy > 1e-10:
                expected_freq *= observed_freq.sum() / expected_freq.sum()
            
            # Chi-square test
            chi_stats = stats.chisquare(observed_freq, f_exp=expected_freq)

            return {
                'chi2_statistic': chi_stats.statistic,
                'p_value': chi_stats.pvalue,
                'n_bins': n_bins,
                'observed_freq': observed_freq,
                'expected_freq': expected_freq
            }
        
        elif method.lower() in ['kolmogorov-smirnov', 'ks']:
            # KS test
            ks_stats = stats.kstest(self.data, self.fitted_distribution.cdf, args=self.fitted_params)
            return {
                'ks_statistic': ks_stats.statistic,
                'p_value': ks_stats.pvalue
            }

        elif method.lower() in ['root-mean-square-error', 'rmse']:
            # RMSE between empirical and theoretical CDF
            sorted_data = np.sort(self.data)
            empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            theoretical_cdf = self.fitted_distribution.cdf(sorted_data, *self.fitted_params)
            rmse = np.sqrt(np.mean((empirical_cdf - theoretical_cdf) ** 2))
            return rmse
        
        elif method.lower() == 'aic':
            # Akaike Information Criterion
            # AIC = 2k - 2ln(L) where k = number of parameters, L = likelihood
            log_likelihood = np.sum(self.fitted_distribution.logpdf(self.data, *self.fitted_params))
            k = len(self.fitted_params)
            aic = 2 * k - 2 * log_likelihood
            return aic
        
        elif method.lower() == 'bic':
            # Bayesian Information Criterion
            # BIC = k*ln(n) - 2ln(L) where k = params, n = sample size, L = likelihood
            log_likelihood = np.sum(self.fitted_distribution.logpdf(self.data, *self.fitted_params))
            k = len(self.fitted_params)
            n = len(self.data)
            bic = k * np.log(n) - 2 * log_likelihood
            return bic
        
        else:
            raise ValueError(f"Unknown goodness-of-fit method: {method}. "
                           f"Available: 'chi2', 'ks', 'rmse', 'aic', 'bic'")


    def _generate_subsample_indices(
        self,
        size: int,
        n_repeats: int,
        sampling: str = 'random',
        seed: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Generate index arrays for subsamples according to a chosen strategy.

        Parameters
        ----------
        size : int
            Number of elements in each subsample.
        n_repeats : int
            Number of subsamples (repeats) to generate per size.
        sampling : {'random','bootstrap','disjoint'}, optional
            Sampling strategy. Defaults to 'random'.
        seed : int, optional
            Seed for reproducible RNG. If None, RNG will be random.

        Returns
        -------
        List[numpy.ndarray]
            A list with `n_subsamples` index arrays (dtype int) pointing into ``self.data``.

        Notes
        -----
        - This function returns index arrays (views) and does not copy the data itself.
        - 'random' = sampling without replacement (size must be <= len(data)).
        - 'bootstrap' = sampling with replacement (allows size > len(data)).
        - 'disjoint' = non-overlapping partitions; will raise if size > len(data).
        """
        # local import to keep top-level imports minimal
        rng = np.random.default_rng(seed)

        N = len(self.data)
        if sampling not in {'random', 'bootstrap', 'disjoint'}:
            raise ValueError(f"Unknown sampling strategy: {sampling}")

        if sampling == 'random' and size > N:
            raise ValueError("For 'random' sampling size must be <= len(data). Use 'bootstrap' to allow size > N.")

        indices: List[np.ndarray] = []

        if sampling == 'bootstrap':
            # with replacement
            for _ in range(n_repeats):
                idx = rng.integers(0, N, size=size)
                indices.append(idx)

        elif sampling == 'random':
            # without replacement
            for _ in range(n_repeats):
                idx = rng.choice(N, size=size, replace=False)
                indices.append(idx)

        else:  # disjoint
            if size <= 0:
                raise ValueError("size must be > 0 for disjoint sampling")
            per_pass = N // size
            if per_pass == 0:
                raise ValueError("disjoint sampling not possible: size > len(data)")

            subsamples_needed = n_repeats
            while subsamples_needed > 0:
                perm = rng.permutation(N)
                for i in range(per_pass):
                    if subsamples_needed == 0:
                        break
                    start = i * size
                    idx = perm[start:start + size]
                    indices.append(idx)
                    subsamples_needed -= 1

        return indices

    def monte_carlo_fit(
        self,
        sizes: Optional[List[int]] = None,
        n_repeats: int = 20,
        tests: List[str] = ['ks'],
        stability_method: str = 'kneedle',
        fig_output_path: Optional[str] = None,
        plot_type: str = 'series',
        sampling: str = 'random',
        seed: Optional[int] = None,
        min_size: int = 50,
        max_size: Optional[int] = None,
        n_sizes: int = 10,
        distribution_params: Optional[Tuple] = None,
        **kwargs
    ) -> 'xr.Dataset':
        """
        Perform Monte Carlo stability analysis for distribution fitting.
        
        This method evaluates how stable distribution parameters and goodness-of-fit
        statistics are across different sample sizes, helping determine the minimum
        sample size needed for reliable parameter estimation.

        Parameters
        ----------
        sizes : list[int], optional
            Explicit list of sample sizes to test. If None, a grid of sizes
            between min_size and max_size is created.
        n_repeats : int, default=20
            Number of subsamples (repetitions) for each size.
        tests : list[str], default=['ks']
            Goodness-of-fit tests to perform. Options: 'ks', 'chi2', 'rmse'.
            **Recommendation**: Always include 'rmse' for best stability detection.
        stability_method : str, default='cv'
            Method for detecting stability points. Options:
            - 'cv': Coefficient of variation approach (default, robust)
            - 'kneedle': Kneedle algorithm on median curves (best for RMSE)
            - 'plateau': Relative gain heuristic (good for converging parameters)
            - 'aggregate': Median-based detection with tolerance windows (legacy)
            - 'detect': Alias for 'cv' (backward compatibility)
            - None or 'none': Skip stability detection
        fig_output_path : str, optional
            When provided, a 2x3 summary figure is generated and saved to this
            path. The Dataset attribute 'figure_path' will contain the path.
            If None (default) no figure is created (saves time in large runs).
        plot_type : {'series','boxplots'}, default='series'
            Visual style per panel. 'series' plots medians with IQR shading;
            'boxplots' draws boxplots per size.
        sampling : {'random','bootstrap','disjoint'}, default='random'
            Sampling strategy for subsamples.
        seed : int, optional
            Random seed for reproducibility (propagated to subsamples).
        min_size : int, default=50
            Minimum sample size when auto-generating sizes.
        max_size : int, optional
            Maximum sample size. Defaults to full dataset if None.
        n_sizes : int, default=10
            Number of sizes in automatically generated grid.
        distribution_params : tuple, optional
            If provided, these fixed parameters are used for all subsamples
            (no refitting). Use only for scenarios evaluating GOF with known
            parameters. If None (default), each subsample is refitted.
        **kwargs
            Additional parameters:
            
            - bins : int or str, default='doane'
                Binning for chi-square test.
            - fit_kwargs : dict
                Passed to fit_distribution for parameter constraints (e.g.,
                fit_kwargs={'floc': 0}).
            
            Stability detection parameters (method-specific):
            
            For 'cv' method:
                - window_size : int, default=len(sizes)//4
                    Window size for CV consistency check
                - cv_threshold : float, default=0.1
                    Maximum acceptable coefficient of variation (10%)
            
            For 'kneedle' method:
                - smooth : bool, default=True
                    Whether to smooth curves before detection
                - smoothing_method : str, default='savgol'
                    Smoothing method: 'savgol' or 'spline'
            
            For 'plateau' method:
                - consecutive_points : int, default=3
                    Number of consecutive points for plateau (L parameter)
                - relative_tolerance : float, default=0.01
                    Maximum relative change threshold (Δ parameter, 1%)

        Returns
        -------
        xarray.Dataset
            Monte Carlo results with dimensions:
            
            - **sizes** : Sample sizes tested
            - **repeats** : Repetition index for each size
            
            Data variables include:
            
            - **param_0, param_1, ...** : Fitted distribution parameters
            - **ks_statistic, ks_pvalue** : Kolmogorov-Smirnov test results
            - **chi2_statistic, chi2_pvalue** : Chi-square test results  
            - **rmse** : Root mean square error values
            
            Attributes include:
            
            - **distribution** : Distribution name
            - **original_data_size** : Size of original dataset
            - **sampling_method** : Sampling strategy used
            - **bins_method** : Binning method for chi-square test
            - **stability_method** : Detection method used
            - **stability_points** : Detected stability points per variable
            - **recommended_size** : Recommended sample size (from primary metric)
            - **primary_metric** : Metric used for recommendation (typically 'rmse')
            - **stable_pvalue_ks** : KS p-value at stable size (if applicable)
            - **stable_pvalue_chi2** : Chi-square p-value at stable size (if applicable)
            - **stable_rmse** : RMSE at stable size (if applicable)
            - **figure_path** : Path to saved 2x3 summary figure (only if generated)
            - **created_by** : Method identifier

        Examples
        --------
        >>> # Basic usage with recommended settings (includes RMSE!)
        >>> results = adjuster.monte_carlo_fit(
        ...     n_repeats=50, 
        ...     tests=['ks', 'chi2', 'rmse'],
        ...     stability_method='kneedle'  # Best for RMSE
        ... )
        
        >>> # Access stability information
        >>> stability = results.attrs['stability_points']
        >>> recommended_size = results.attrs['recommended_size']
        >>> print(f"RMSE stabilizes at n={stability['rmse']['size']}")
        
        >>> # Use fit constraints (e.g., fix location for Weibull)
        >>> results = adjuster.monte_carlo_fit(
        ...     n_repeats=100, 
        ...     tests=['rmse', 'ks'],
        ...     stability_method='plateau',
        ...     fit_kwargs={'floc': 0}
        ... )
        
        >>> # Use pre-calculated parameters (bypass fitting entirely)
        >>> known_params = (2.0, 0.0, 1.0)  # shape, loc, scale
        >>> results = adjuster.monte_carlo_fit(
        ...     distribution_params=known_params,
        ...     n_repeats=50,
        ...     tests=['chi2', 'ks']
        ... )
        
        >>> # Generate figure with automatic stability detection
        >>> results = adjuster.monte_carlo_fit(
        ...     tests=['ks', 'rmse'],
        ...     stability_method='kneedle',
        ...     fig_output_path='mc_analysis.png'
        ... )

        Notes
        -----
        **Fitting Strategy:**
        
        By default, each subsample gets an independent fit to evaluate how parameter
        estimates change with sample size. This is the correct approach for stability
        analysis since parameter estimation quality depends on sample size.
        
        **Stability Detection Methods:**
        
        1. **CV (Coefficient of Variation)**: Default method, robust and works for all
           variables. Looks for when std/mean becomes consistently small.
        
        2. **Kneedle**: Best for RMSE and converging parameters. Finds the "elbow" point
           where improvement rate changes. Requires curve smoothing.
        
        3. **Plateau**: Good for metrics with clear convergence. Uses relative gain
           heuristic (Δᵢ = |y(i-1) - y(i)| / |y(i-1)|).
        
        **Choosing the Right Method:**
        
        - Use **'kneedle'** with **RMSE** for clearest stability detection
        - Use **'cv'** for general-purpose analysis
        - Use **'plateau'** when you expect clear convergence behavior
        
        **Parameter Constraints:**
        
        Use `fit_kwargs` to impose constraints during fitting (e.g., `floc=0` for 
        Weibull distributions). These constraints apply to all subsample fits.
        
        **Pre-calculated Parameters:**
        
        Use `distribution_params` only when you want to evaluate goodness-of-fit
        with known/fixed parameters across all sample sizes, bypassing fitting entirely.
        """

        # Extract kwargs
        bins = kwargs.get('bins', 'doane')
        fit_kwargs = kwargs.get('fit_kwargs', {})
        
        N = len(self.data)
        if N == 0:
            raise ValueError("No data available for monte_carlo_fit.")

        # Determine sizes
        if sizes is None:
            if max_size is None:
                max_size = N
            sizes = np.unique(np.linspace(min_size, max_size, n_sizes, dtype=int))
            sizes = sizes[sizes > 0]
            if sizes.size == 0:
                raise ValueError("Generated empty sizes grid; check min_size and n_sizes values.")
        else:
            sizes = np.unique(np.asarray(sizes, dtype=int))
            if sizes.size == 0:
                raise ValueError("`sizes` provided is empty or invalid")

        # Determine repeats per size
        if n_repeats is None:
            smallest = int(sizes[0])
            n_repeats = max(10, min(100, N // smallest))

        test_list = [t.lower() for t in tests] if tests is not None else ['ks']
        
        # Determine max number of parameters
        if distribution_params is not None:
            # Use pre-calculated parameters to determine count
            max_params = len(distribution_params)
        else:
            # Fit a small sample to determine parameter count
            sample_idx = np.random.choice(len(self.data), min(100, len(self.data)), replace=False)
            sample_data = self.data[sample_idx]
            temp_dp = DataProcessor(sample_data)
            temp_adj = temp_dp._get_adjuster()
            temp_adj.fit_distribution(self.fitted_distribution, **fit_kwargs)
            max_params = len(temp_adj.get_fitted_params())

        # Initialize data arrays
        n_sizes = len(sizes)
        param_arrays = {}
        for i in range(max_params):
            param_arrays[f'param_{i}'] = np.full((n_sizes, n_repeats), np.nan)

        # Initialize test result arrays
        test_arrays = {}
        for test in test_list:
            if test == 'ks':
                test_arrays['ks_statistic'] = np.full((n_sizes, n_repeats), np.nan)
                test_arrays['ks_pvalue'] = np.full((n_sizes, n_repeats), np.nan)
            elif test == 'chi2':
                test_arrays['chi2_statistic'] = np.full((n_sizes, n_repeats), np.nan)
                test_arrays['chi2_pvalue'] = np.full((n_sizes, n_repeats), np.nan)
            elif test == 'rmse':
                test_arrays['rmse'] = np.full((n_sizes, n_repeats), np.nan)

        # Setup random seeds
        if seed is None:
            child_seq = [None] * n_sizes
        else:
            ss = np.random.SeedSequence(seed)
            child_seq = ss.spawn(n_sizes)

        # Prepare storage for aggregate-style results if requested
        aggregate_results = {'results': {int(s): [] for s in sizes}} if stability_method == 'aggregate' else None

        # Main Monte Carlo loop
        sizes_iter = tqdm(list(enumerate(sizes)), desc='Monte Carlo sizes')
        for i, size in sizes_iter:
            child_seed = child_seq[i]

            idx_list = self._generate_subsample_indices(
                size=int(size),
                n_repeats=int(n_repeats),
                sampling=sampling,
                seed=child_seed
            )

            # Inner progress bar for repeats
            rep_iter = trange(len(idx_list), desc=f'size={size}', leave=False)
            for rep_j in rep_iter:
                try:
                    sub_idx = idx_list[rep_j]
                    subdata = self.data[sub_idx]

                    # Create temporary adjuster
                    dp = DataProcessor(subdata)
                    adj = dp._get_adjuster()

                    if self.fitted_distribution is None:
                        raise ValueError("No fitted distribution available.")

                    # Always fit new parameters for each subsample
                    # Use distribution_params only if explicitly provided (for pre-calculated scenarios)
                    if distribution_params is not None:
                        # Use pre-calculated parameters (bypass fitting)
                        adj.fitted_distribution = self.fitted_distribution
                        adj.fitted_params = distribution_params
                        params = distribution_params
                    else:
                        # Fit new parameters with optional constraints from fit_kwargs
                        adj.fit_distribution(self.fitted_distribution, **fit_kwargs)
                        params = adj.get_fitted_params()

                    # Store parameters
                    for param_idx, param_val in enumerate(params):
                        if param_idx < max_params:
                            param_arrays[f'param_{param_idx}'][i, rep_j] = param_val

                    # If aggregate collection requested, store per-rep entry
                    if aggregate_results is not None:
                        rep_entry = {'params': params, 'gof': {}}

                    # Run goodness-of-fit tests
                    for test in test_list:
                        try:
                            if test == 'ks':
                                ks_res = adj.goodness_of_fit('ks')
                                test_arrays['ks_statistic'][i, rep_j] = ks_res.get('statistic', ks_res.get('ks_statistic', np.nan))
                                test_arrays['ks_pvalue'][i, rep_j] = ks_res.get('p_value', np.nan)
                                if aggregate_results is not None:
                                    rep_entry['gof']['ks'] = ks_res
                            elif test == 'chi2':
                                chi_res = adj.goodness_of_fit('chi2', bins=bins, warn_on_normalization=False)
                                test_arrays['chi2_statistic'][i, rep_j] = chi_res.get('statistic', chi_res.get('chi2_statistic', np.nan))
                                test_arrays['chi2_pvalue'][i, rep_j] = chi_res.get('p_value', np.nan)
                                if aggregate_results is not None:
                                    rep_entry['gof']['chi2'] = chi_res
                            elif test == 'rmse':
                                test_arrays['rmse'][i, rep_j] = adj.goodness_of_fit('rmse')
                                if aggregate_results is not None:
                                    rep_entry['gof']['rmse'] = test_arrays['rmse'][i, rep_j]
                        except Exception:
                            # Values remain NaN for failed tests
                            pass

                    if aggregate_results is not None:
                        aggregate_results['results'][int(size)].append(rep_entry)

                except Exception:
                    # Values remain NaN for failed fits
                    pass

        # Create xarray Dataset
        data_vars = {}
        data_vars.update(param_arrays)
        data_vars.update(test_arrays)

        coords = {
            'sizes': sizes,
            'repeats': np.arange(n_repeats)
        }

        # Detect stability points using selected method
        stability_points = None
        agg_summary = None
        
        # Normalize stability_method name
        if stability_method is None or stability_method.lower() == 'none':
            # Skip stability detection
            stability_points = {k: {'size': None, 'index': None, 'cv_at_stability': None, 'smoothed_curve': None, 'method': 'none'} 
                              for k in list(param_arrays.keys()) + list(test_arrays.keys())}
        elif stability_method.lower() == 'detect':
            # Backward compatibility: 'detect' maps to 'cv'
            stability_points = self._detect_stability_unified(data_vars, sizes, method='cv', **kwargs)
        elif stability_method.lower() in ['cv', 'kneedle', 'plateau']:
            # Use unified detection with specified method
            stability_points = self._detect_stability_unified(data_vars, sizes, method=stability_method.lower(), **kwargs)
        elif stability_method.lower() == 'aggregate' and aggregate_results is not None:
            # Legacy aggregate method
            agg_summary, _, _ = self._aggregate_and_detect_stability(aggregate_results, sizes, test_list, max_params)
            # Convert agg_summary to stability_points format used by plotting
            stability_points = {}
            # Tests
            for test in agg_summary.get('tests', {}):
                inf_idx = agg_summary['tests'][test].get('inflection_index')
                stability_points[test] = {
                    'size': int(sizes[inf_idx]) if inf_idx is not None else None,
                    'index': int(inf_idx) if inf_idx is not None else None,
                    'cv_at_stability': None,
                    'smoothed_curve': None,
                    'method': 'aggregate'
                }
            # Params (use params inflection)
            p_inf_idx = agg_summary['params'].get('inflection_index')
            stability_points['param_0'] = {
                'size': int(sizes[p_inf_idx]) if p_inf_idx is not None else None,
                'index': int(p_inf_idx) if p_inf_idx is not None else None,
                'cv_at_stability': None,
                'smoothed_curve': None,
                'method': 'aggregate'
            }
        else:
            raise ValueError(f"Unknown stability_method: {stability_method}. "
                           f"Options: 'cv', 'kneedle', 'plateau', 'aggregate', 'detect', 'none'")
        
        # Determine recommended size and primary metric
        # Priority: RMSE > KS > Chi2 > param_0
        recommended_size = None
        primary_metric = None
        
        for metric in ['rmse', 'ks_pvalue', 'chi2_pvalue', 'param_0']:
            if metric in stability_points and stability_points[metric]['size'] is not None:
                recommended_size = stability_points[metric]['size']
                primary_metric = metric
                break
        
        # If still no recommendation, use largest size
        if recommended_size is None:
            recommended_size = int(sizes[-1])
            primary_metric = 'max_size'
        
        # Get p-values and metrics at stable size for reporting
        stable_idx = None
        for metric_name, info in stability_points.items():
            if info['size'] == recommended_size:
                stable_idx = info['index']
                break
        
        stable_pvalue_ks = None
        stable_pvalue_chi2 = None
        stable_rmse = None
        
        if stable_idx is not None:
            if 'ks_pvalue' in data_vars:
                ks_at_stable = data_vars['ks_pvalue'][stable_idx, :]
                stable_pvalue_ks = float(np.nanmedian(ks_at_stable))
            if 'chi2_pvalue' in data_vars:
                chi2_at_stable = data_vars['chi2_pvalue'][stable_idx, :]
                stable_pvalue_chi2 = float(np.nanmedian(chi2_at_stable))
            if 'rmse' in data_vars:
                rmse_at_stable = data_vars['rmse'][stable_idx, :]
                stable_rmse = float(np.nanmedian(rmse_at_stable))
        
        # Optionally create & save figure
        figure_path = None
        if fig_output_path is not None:
            try:
                fig = self._create_monte_carlo_figure(
                    data_vars, sizes, plot_type, stability_points,
                    distribution_name=self.distribution_name or str(self.fitted_distribution),
                    max_size=int(sizes[-1]),
                    recommended_size=recommended_size,
                    primary_metric=primary_metric,
                    stable_pvalue_ks=stable_pvalue_ks,
                    stable_pvalue_chi2=stable_pvalue_chi2,
                    stable_rmse=stable_rmse,
                    sampling_method=sampling,
                    stability_method=stability_method
                )
                if fig is not None:
                    fig.savefig(fig_output_path, dpi=150, bbox_inches='tight')
                    figure_path = fig_output_path
            except Exception as e:
                warnings.warn(f"Failed to create figure: {e}")
                figure_path = None

        # Create Dataset (all numeric data only)
        ds = xr.Dataset(
            data_vars={name: (['sizes', 'repeats'], array) for name, array in data_vars.items()},
            coords=coords,
            attrs={
                'distribution': self.distribution_name or str(self.fitted_distribution),
                'original_data_size': N,
                'sampling_method': sampling,
                'bins_method': bins,
                'stability_method': stability_method,
                'stability_points': stability_points,
                'recommended_size': recommended_size,
                'primary_metric': primary_metric,
                'stable_pvalue_ks': stable_pvalue_ks,
                'stable_pvalue_chi2': stable_pvalue_chi2,
                'stable_rmse': stable_rmse,
                'n_repeats': n_repeats,
                'min_size': int(sizes[0]),
                'max_size': int(sizes[-1]),
                'n_sizes': len(sizes),
                'figure_path': figure_path,
                'created_by': 'MagicAdjuster.monte_carlo_fit'
            }
        )

        return ds

    def _smooth_curve(self, x: np.ndarray, y: np.ndarray, method: str = 'savgol') -> np.ndarray:
        """
        Smooth a curve for stability detection.
        
        Parameters
        ----------
        x : np.ndarray
            X values (sample sizes)
        y : np.ndarray
            Y values (metric to smooth)
        method : str, default='savgol'
            Smoothing method: 'savgol' (Savitzky-Golay) or 'spline'
            
        Returns
        -------
        np.ndarray
            Smoothed y values
            
        Notes
        -----
        Savitzky-Golay filter is preferred for preserving features while
        smoothing noise. Spline interpolation provides smoother curves but
        may over-smooth important features.
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < 4:
            return y  # Not enough points to smooth
            
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if method == 'savgol':
            # Savitzky-Golay filter - preserves features
            window_length = min(len(x_valid), max(5, len(x_valid) // 3))
            if window_length % 2 == 0:
                window_length -= 1  # Must be odd
            poly_order = min(3, window_length - 1)
            
            try:
                y_smooth = savgol_filter(y_valid, window_length, poly_order)
                # Map back to original x positions
                result = np.full_like(y, np.nan)
                result[valid_mask] = y_smooth
                return result
            except:
                return y
                
        elif method == 'spline':
            # Spline interpolation - smoother but may over-smooth
            try:
                # Use spline with automatic smoothing
                spl = UnivariateSpline(x_valid, y_valid, s=len(x_valid) * 0.1, k=3)
                result = np.full_like(y, np.nan)
                result[valid_mask] = spl(x_valid)
                return result
            except:
                return y
        else:
            return y

    def _kneedle_detection(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        smooth: bool = True,
        smoothing_method: str = 'savgol'
    ) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """
        Detect elbow/knee point using the Kneedle algorithm.
        
        Based on Satopää et al. (2011): "Finding a 'Kneedle' in a Haystack:
        Detecting Knee Points in System Behavior"
        
        Parameters
        ----------
        x : np.ndarray
            Sample sizes (must be monotonic)
        y : np.ndarray
            Metric values (e.g., RMSE, median parameter)
        smooth : bool, default=True
            Whether to smooth the curve before detection
        smoothing_method : str, default='savgol'
            Method for smoothing: 'savgol' or 'spline'
            
        Returns
        -------
        knee_idx : int or None
            Index of detected knee point, or None if not found
        y_smooth : np.ndarray or None
            Smoothed curve if smoothing was applied, else None
            
        Notes
        -----
        The algorithm:
        1. Normalizes x and y to [0,1]
        2. Computes difference between curve and reference line
        3. Finds maximum difference (the "knee")
        
        Works best for decreasing metrics (RMSE) or converging parameters.
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < 3:
            return None, None
            
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # Smooth if requested
        y_smooth = None
        if smooth:
            y_work = self._smooth_curve(x, y, method=smoothing_method)
            y_work_valid = y_work[valid_mask]
            y_smooth = y_work
        else:
            y_work_valid = y_valid
        
        # Normalize to [0, 1]
        x_norm = (x_valid - x_valid.min()) / (x_valid.max() - x_valid.min() + 1e-10)
        y_norm = (y_work_valid - y_work_valid.min()) / (y_work_valid.max() - y_work_valid.min() + 1e-10)
        
        # Calculate reference line (straight line from start to end)
        y_ref = np.linspace(y_norm[0], y_norm[-1], len(y_norm))
        
        # Calculate differences (for decreasing curves, use y_ref - y_norm)
        # For increasing curves, use y_norm - y_ref
        # Auto-detect curve direction
        if y_norm[-1] < y_norm[0]:
            # Decreasing curve (e.g., RMSE)
            differences = y_ref - y_norm
        else:
            # Increasing curve (e.g., p-values sometimes)
            differences = y_norm - y_ref
        
        # Find maximum difference (the knee)
        if len(differences) == 0:
            return None, y_smooth
            
        knee_idx_local = np.argmax(differences)
        
        # Map back to original indices
        valid_indices = np.where(valid_mask)[0]
        knee_idx = valid_indices[knee_idx_local]
        
        return int(knee_idx), y_smooth

    def _plateau_detection(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        consecutive_points: int = 3,
        relative_tolerance: float = 0.01
    ) -> Optional[int]:
        """
        Detect plateau using relative gain heuristic.
        
        Based on the concept that stability is reached when the relative
        improvement between consecutive sample sizes becomes negligible.
        
        Parameters
        ----------
        x : np.ndarray
            Sample sizes
        y : np.ndarray  
            Metric values (medians across repeats)
        consecutive_points : int, default=3
            Number of consecutive points that must satisfy the criterion
            (L parameter in the heuristic formula)
        relative_tolerance : float, default=0.01
            Maximum acceptable relative change (Δ parameter, default 1%)
            
        Returns
        -------
        plateau_idx : int or None
            Index where plateau starts, or None if not found
            
        Notes
        -----
        The algorithm computes relative gain:
            Δᵢ = |RMSE(nᵢ₋₁) - RMSE(nᵢ)| / RMSE(nᵢ₋₁)
        
        Plateau is detected when Δᵢ < threshold for L consecutive points.
        
        Works best for decreasing metrics (RMSE) or parameter convergence.
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < consecutive_points + 1:
            return None
            
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # Calculate relative changes
        rel_changes = []
        for i in range(1, len(y_valid)):
            if y_valid[i-1] != 0:
                rel_change = abs(y_valid[i] - y_valid[i-1]) / abs(y_valid[i-1])
                rel_changes.append(rel_change)
            else:
                rel_changes.append(np.inf)
        
        # Find first point where relative changes stay below tolerance
        # for consecutive_points consecutive measurements
        for i in range(len(rel_changes) - consecutive_points + 1):
            window = rel_changes[i:i + consecutive_points]
            if all(rc < relative_tolerance for rc in window if not np.isinf(rc)):
                # Map back to original indices
                valid_indices = np.where(valid_mask)[0]
                # Return index after which the plateau starts (i+1 because we skip first point)
                plateau_idx = valid_indices[i + 1]
                return int(plateau_idx)
        
        return None

    def _detect_stability_unified(
        self,
        data_vars: Dict[str, np.ndarray],
        sizes: np.ndarray,
        method: str = 'cv',
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Unified stability detection supporting multiple methods.
        
        Parameters
        ----------
        data_vars : dict
            Dictionary of variable names to (sizes x repeats) arrays
        sizes : np.ndarray
            Sample sizes tested
        method : str, default='cv'
            Detection method:
            - 'cv': Coefficient of variation (original method)
            - 'kneedle': Kneedle algorithm on median curve
            - 'plateau': Relative gain heuristic on median curve
        **kwargs
            Method-specific parameters:
            - For 'cv': window_size, cv_threshold
            - For 'kneedle': smooth, smoothing_method
            - For 'plateau': consecutive_points, relative_tolerance
            
        Returns
        -------
        dict
            Stability points for each variable with keys:
            - 'size': Sample size where stability detected (or None)
            - 'index': Index in sizes array (or None)
            - 'cv_at_stability': CV value if method='cv' (or None)
            - 'smoothed_curve': Smoothed median curve if applicable
            - 'method': Detection method used
            
        Notes
        -----
        Different methods work better for different metrics:
        - 'cv': General purpose, works for all variables
        - 'kneedle': Best for decreasing metrics (RMSE) or converging parameters
        - 'plateau': Best for metrics with clear convergence plateaus
        """
        stability_points = {}
        
        # Extract kwargs for each method
        # CV method
        window_size = kwargs.get('window_size', max(2, len(sizes) // 4))
        cv_threshold = kwargs.get('cv_threshold', 0.1)
        
        # Kneedle method
        smooth = kwargs.get('smooth', True)
        smoothing_method = kwargs.get('smoothing_method', 'savgol')
        
        # Plateau method
        consecutive_points = kwargs.get('consecutive_points', 3)
        relative_tolerance = kwargs.get('relative_tolerance', 0.01)
        
        for var_name, data_array in data_vars.items():
            if np.all(np.isnan(data_array)):
                stability_points[var_name] = {
                    'size': None,
                    'index': None,
                    'cv_at_stability': None,
                    'smoothed_curve': None,
                    'method': method
                }
                continue
            
            # Calculate medians for methods that need them
            medians = np.array([np.nanmedian(data_array[i, :]) for i in range(len(sizes))])
            
            stable_idx = None
            smoothed_curve = None
            cv_at_stability = None
            
            if method == 'cv':
                # Original CV-based method
                cv_values = []
                for i in range(len(sizes)):
                    size_data = data_array[i, :]
                    valid_data = size_data[~np.isnan(size_data)]
                    if len(valid_data) > 1:
                        mean_val = np.mean(valid_data)
                        std_val = np.std(valid_data)
                        cv = std_val / abs(mean_val) if mean_val != 0 else np.inf
                        cv_values.append(cv)
                    else:
                        cv_values.append(np.inf)
                
                # Find first point where CV stays below threshold
                for i in range(len(cv_values) - window_size + 1):
                    window_cvs = cv_values[i:i + window_size]
                    if all(cv < cv_threshold for cv in window_cvs if not np.isinf(cv)):
                        stable_idx = i
                        cv_at_stability = cv_values[i]
                        break
            
            elif method == 'kneedle':
                # Kneedle algorithm on median curve
                stable_idx, smoothed_curve = self._kneedle_detection(
                    sizes, medians, smooth=smooth, smoothing_method=smoothing_method
                )
            
            elif method == 'plateau':
                # Plateau detection on median curve
                stable_idx = self._plateau_detection(
                    sizes, medians,
                    consecutive_points=consecutive_points,
                    relative_tolerance=relative_tolerance
                )
            
            else:
                raise ValueError(f"Unknown stability detection method: {method}")
            
            # Store results
            if stable_idx is not None:
                stability_points[var_name] = {
                    'size': int(sizes[stable_idx]),
                    'index': int(stable_idx),
                    'cv_at_stability': cv_at_stability,
                    'smoothed_curve': smoothed_curve,
                    'method': method
                }
            else:
                stability_points[var_name] = {
                    'size': None,
                    'index': None,
                    'cv_at_stability': None,
                    'smoothed_curve': smoothed_curve,
                    'method': method
                }
        
        return stability_points

    def _detect_stability_points(self, data_vars: Dict[str, np.ndarray], sizes: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Detect stability points for each variable using coefficient of variation.
        
        Legacy method kept for backward compatibility. Use _detect_stability_unified instead.
        
        A variable is considered stable when its coefficient of variation
        (std/mean) across repeats becomes consistently low.
        """
        return self._detect_stability_unified(data_vars, sizes, method='cv')

    def _aggregate_and_detect_stability(
        self,
        results: Dict[str, Any],
        sizes: np.ndarray,
        test_list: Optional[List[str]] = None,
        max_params: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[int, List[List[float]]], Dict[int, List[float]]]:
        """
        Aggregate monte carlo `results` and detect stability (CPS) for tests and parameters.

        Returns (summary, param_values_per_size, param_medians)
        """
        summary: Dict[str, Any] = {'sizes': sizes.tolist(), 'tests': {}, 'params': {}}

        # If test_list not provided, infer from first non-empty rep
        if test_list is None:
            inferred = set()
            for size in sizes:
                reps = results.get('results', {}).get(int(size), [])
                if len(reps) > 0:
                    for rep in reps:
                        gof = rep.get('gof', {})
                        inferred.update(gof.keys())
                    break
            test_list = sorted(list(inferred)) if inferred else []

        # If max_params not provided, infer from first available params
        if max_params is None:
            inferred_max = 0
            for size in sizes:
                reps = results.get('results', {}).get(int(size), [])
                for rep in reps:
                    params = rep.get('params')
                    if params is not None:
                        inferred_max = max(inferred_max, len(params))
                if inferred_max > 0:
                    break
            max_params = int(inferred_max)

        # For each test, compute list of p-values (or metric) per size
        for test in test_list:
            values_per_size = []
            for size in sizes:
                reps = results['results'].get(int(size), [])
                vals = []
                for rep in reps:
                    gof = rep.get('gof', {})
                    entry = gof.get(test)
                    if entry is None:
                        continue
                    # chi2 and ks return dict with 'p_value', rmse returns scalar
                    if isinstance(entry, dict):
                        p = entry.get('p_value') if 'p_value' in entry else None
                        if p is not None:
                            vals.append(float(p))
                        else:
                            stat = entry.get('chi2_statistic') or entry.get('ks_statistic')
                            if stat is not None:
                                vals.append(float(stat))
                    else:
                        try:
                            vals.append(float(entry))
                        except Exception:
                            continue
                values_per_size.append(vals)

            # median per size
            medians = np.array([np.median(v) if len(v) > 0 else np.nan for v in values_per_size])

            # detect inflection / stability: look for first index where moving window is stable
            tol = 0.1
            window = 4
            inflection_idx = None
            if medians.size >= window:
                for j in range(0, len(medians) - window + 1):
                    window_vals = medians[j:j+window]
                    if np.all(np.isfinite(window_vals)) and (np.nanmax(window_vals) - np.nanmin(window_vals) <= tol):
                        inflection_idx = j + (window - 1)
                        break

            inflection_size = int(sizes[inflection_idx]) if inflection_idx is not None else None

            summary['tests'][test] = {
                'values_per_size': values_per_size,
                'medians': medians.tolist(),
                'inflection_index': int(inflection_idx) if inflection_idx is not None else None,
                'inflection_size': inflection_size,
            }

        # Aggregate parameters: collect per-parameter lists per size and compute medians
        param_medians = {p: [] for p in range(max_params)}
        param_values_per_size = {p: [] for p in range(max_params)}
        for size in sizes:
            reps = results['results'].get(int(size), [])
            cols = {p: [] for p in range(max_params)}
            for rep in reps:
                params = rep.get('params')
                if params is None:
                    continue
                for p in range(max_params):
                    if p < len(params):
                        try:
                            cols[p].append(float(params[p]))
                        except Exception:
                            pass
            for p in range(max_params):
                param_values_per_size[p].append(cols[p])
                param_medians[p].append(np.median(cols[p]) if len(cols[p]) > 0 else np.nan)

        summary['params']['values_per_size'] = param_values_per_size
        summary['params']['medians'] = {p: np.array(v).tolist() for p, v in param_medians.items()}

        # Detect inflection for parameters (use first parameter as representative)
        if max_params > 0:
            p0_meds = np.array(param_medians[0])
            tol_p = 1e-3 * (np.nanmax(p0_meds) - np.nanmin(p0_meds) if np.nanmax(p0_meds) != np.nanmin(p0_meds) else 1.0)
            inflection_idx_p = None
            if p0_meds.size >= window:
                for j in range(0, len(p0_meds) - window + 1):
                    window_vals = p0_meds[j:j+window]
                    if np.all(np.isfinite(window_vals)) and (np.nanmax(window_vals) - np.nanmin(window_vals) <= tol_p):
                        inflection_idx_p = j + (window - 1)
                        break
            summary['params']['inflection_index'] = int(inflection_idx_p) if inflection_idx_p is not None else None
            summary['params']['inflection_size'] = int(sizes[inflection_idx_p]) if inflection_idx_p is not None else None
        else:
            summary['params']['inflection_index'] = None
            summary['params']['inflection_size'] = None

        return summary, param_values_per_size, param_medians
    

    def _create_monte_carlo_figure(
        self,
        data_vars: Dict[str, np.ndarray],
        sizes: np.ndarray,
        plot_type: str,
        stability_points: Optional[Dict[str, Dict[str, Any]]] = None,
        distribution_name: Optional[str] = None,
        max_size: Optional[int] = None,
        recommended_size: Optional[int] = None,
        primary_metric: Optional[str] = None,
        stable_pvalue_ks: Optional[float] = None,
        stable_pvalue_chi2: Optional[float] = None,
        stable_rmse: Optional[float] = None,
        sampling_method: Optional[str] = None,
        stability_method: Optional[str] = None,
    ):
        """
        Create 2x3 summary figure with enhanced information.
        
        Parameters
        ----------
        data_vars : dict
            Dictionary of variable names to (sizes x repeats) arrays
        sizes : np.ndarray
            Sample sizes tested
        plot_type : str
            'series' or 'boxplots'
        stability_points : dict, optional
            Stability information for each variable
        distribution_name : str, optional
            Name of fitted distribution
        max_size : int, optional
            Maximum sample size tested
        recommended_size : int, optional
            Recommended sample size from stability analysis
        primary_metric : str, optional
            Primary metric used for recommendation
        stable_pvalue_ks : float, optional
            KS p-value at stable size
        stable_pvalue_chi2 : float, optional
            Chi-square p-value at stable size
        stable_rmse : float, optional
            RMSE at stable size
            
        Returns
        -------
        matplotlib.figure.Figure or None
            Generated figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            return None

        # Select up to first 3 parameter variables
        param_names = sorted([k for k in data_vars if k.startswith('param_')])[:3]
        if not param_names:
            return None  # Nothing meaningful to show

        # Preferred order for test panels (only keep those present)
        preferred_tests = ["ks_pvalue", "chi2_pvalue", "rmse"]
        test_names = [t for t in preferred_tests if t in data_vars][:3]

        # Always build a 2x3 grid; hide unused axes
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        axes_flat = axes.ravel()

        def _panel(ax, data: np.ndarray, title: str, var_name: str):
            """Create individual panel with data and stability markers."""
            # Plot main data
            if plot_type == 'boxplots':
                box_data = [data[i, ~np.isnan(data[i, :])] for i in range(len(sizes))]
                if len(sizes) > 1:
                    width = max(1, (sizes[1] - sizes[0]) * 0.6)
                else:
                    width = 5
                bp = ax.boxplot(box_data, positions=sizes, widths=width, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
            else:  # series
                med = np.nanmedian(data, axis=1)
                q25 = np.nanpercentile(data, 25, axis=1)
                q75 = np.nanpercentile(data, 75, axis=1)
                ax.plot(sizes, med, 'o-', lw=2, label='Median', color='steelblue', markersize=5)
                ax.fill_between(sizes, q25, q75, alpha=0.3, label='IQR (25-75%)', color='steelblue')
                
                # Add smoothed curve if available
                if stability_points and var_name in stability_points:
                    sp = stability_points[var_name]
                    if sp.get('smoothed_curve') is not None:
                        smoothed = sp['smoothed_curve']
                        valid_mask = ~np.isnan(smoothed)
                        if np.sum(valid_mask) > 0:
                            ax.plot(sizes[valid_mask], smoothed[valid_mask], 
                                  '--', lw=1.5, label='Smoothed', color='orange', alpha=0.8)
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Stability vertical lines
            legend_items = []
            
            # Red line: primary metric stability point (shown in all panels)
            if recommended_size is not None:
                try:
                    is_primary = (var_name == primary_metric)
                    label = f"Stable: n={recommended_size}" if is_primary else f"RMSE stable: n={recommended_size}"
                    vline_red = ax.axvline(recommended_size, color='red', linestyle='--', 
                                          linewidth=2, alpha=0.7, zorder=10)
                    legend_items.append((vline_red, label))
                except Exception:
                    pass
            
            # Gray line: this metric's own stability point (if different from primary)
            if stability_points and var_name in stability_points and var_name != primary_metric:
                sp = stability_points[var_name]
                if sp and sp.get('size') is not None and sp['size'] != recommended_size:
                    try:
                        vline_gray = ax.axvline(sp['size'], color='gray', linestyle=':', 
                                              linewidth=1.5, alpha=0.6, zorder=9)
                        legend_items.append((vline_gray, f"{var_name} stable: n={sp['size']}"))
                    except Exception:
                        pass

            # Draw horizontal alpha threshold for p-value panels (KS / Chi2)
            if var_name in ('ks_pvalue', 'chi2_pvalue'):
                try:
                    hline = ax.axhline(0.05, color='darkred', linestyle=':', 
                                      linewidth=1.5, alpha=0.7, zorder=9)
                    legend_items.append((hline, 'α = 0.05'))
                    
                    # Add p-value at stable point if available
                    if var_name == 'ks_pvalue' and stable_pvalue_ks is not None:
                        ax.text(0.98, 0.98, f'p-value @ stable: {stable_pvalue_ks:.3f}',
                               transform=ax.transAxes, fontsize=9,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    elif var_name == 'chi2_pvalue' and stable_pvalue_chi2 is not None:
                        ax.text(0.98, 0.98, f'p-value @ stable: {stable_pvalue_chi2:.3f}',
                               transform=ax.transAxes, fontsize=9,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except Exception:
                    pass
            
            # Add RMSE value at stable point
            if var_name == 'rmse' and stable_rmse is not None:
                try:
                    ax.text(0.98, 0.98, f'RMSE @ stable: {stable_rmse:.4f}',
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                except Exception:
                    pass
            
            # Create legend
            if plot_type == 'series' or legend_items:
                handles, labels = ax.get_legend_handles_labels()
                for item_handle, item_label in legend_items:
                    handles.append(item_handle)
                    labels.append(item_label)
                if handles:
                    ax.legend(handles, labels, frameon=True, fontsize=8, 
                            loc='best', framealpha=0.9)

        # Parameter panels (row 0)
        for col, pname in enumerate(param_names):
            ax = axes[0, col]
            _panel(ax, data_vars[pname], pname.replace('_', ' ').title(), pname)
            ax.set_ylabel('Parameter Value', fontsize=10)
        # Hide unused param axes in row 0
        for col in range(len(param_names), 3):
            axes[0, col].axis('off')

        # Test panels (row 1)
        for col, tname in enumerate(test_names):
            ax = axes[1, col]
            label = tname.replace('_', ' ').title()
            _panel(ax, data_vars[tname], label, tname)
            ax.set_xlabel('Sample Size (n)', fontsize=10)
            if col == 0:
                ax.set_ylabel('Test Value', fontsize=10)
            # Ensure p-value panels show full range [0, 1]
            if 'pvalue' in tname:
                try:
                    ax.set_ylim(-0.05, 1.05)
                except Exception:
                    pass
            # RMSE should start at 0
            elif tname == 'rmse':
                try:
                    ax.set_ylim(bottom=0)
                except Exception:
                    pass
        for col in range(len(test_names), 3):
            axes[1, col].axis('off')

        # Shared formatting
        for ax in axes_flat:
            if ax.has_data():
                ax.tick_params(axis='both', labelsize=9)

        # Create informative title
        title_parts = ['Monte Carlo Stability Analysis']
        if distribution_name:
            title_parts.append(f'Distribution: {distribution_name}')
        if max_size:
            title_parts.append(f'Max n: {max_size}')
        if sampling_method:
            title_parts.append(f'Sampling: {sampling_method}')
        if stability_method:
            method_display = stability_method.capitalize()
            title_parts.append(f'Method: {method_display}')
        if recommended_size and primary_metric:
            metric_display = primary_metric.replace('_', ' ').upper()
            title_parts.append(f'Stable @ n={recommended_size} ({metric_display})')
        
        title = ' | '.join(title_parts)
        fig.suptitle(title, fontsize=13, fontweight='bold')
        
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

## TO-DO:
# When set a new distribution to an already used variable, it will override the distribution form previous variable
# For example:
# fitted_data_weibull = data.fit_distribution('weibull')
# fitted_data_norm = data.fit_distribution('norm')
#
# The distribution from "fitted_data_weibull" will return the norm dist.

 