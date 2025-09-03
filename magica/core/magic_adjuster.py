"""
Statistical distribution fitting and adjustment for wind data
"""

from matplotlib.pylab import seed
import numpy as np
from scipy import stats
import xarray as xr
from typing import Union, Dict, Any, Optional, Tuple, List
import warnings
from tqdm import tqdm, trange

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
                # If no positional arguments provided, use original data with fitted params
                if len(args) == 0 and name in ['pdf', 'cdf', 'sf', 'logpdf', 'logcdf', 'logsf']:
                    return original_method(self.data, *self.fitted_params, **kwargs)
                # If only data provided (1 arg), add fitted parameters
                elif len(args) == 1 and name in ['pdf', 'cdf', 'sf', 'logpdf', 'logcdf', 'logsf']:
                    return original_method(args[0], *self.fitted_params, **kwargs)
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
                Options:
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
            # RMSE
            n_bins = self.get_num_bins(bins)
            observed_freq, bin_edges = np.histogram(self.data, bins=n_bins, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            estimated_pdf = self.fitted_distribution.pdf(bin_centers, *self.fitted_params)
            rmse = np.sqrt(np.mean((observed_freq - estimated_pdf) ** 2))
            return rmse


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
            - bins : int or str, default='doane'
                Binning for chi-square test.
            - fit_kwargs : dict
                Passed to fit_distribution for parameter constraints (e.g.,
                fit_kwargs={'floc': 0}).

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
            - **stability_points** : Detected stability points per variable
            - **figure_path** : Path to saved 2x3 summary figure (only if generated)

        Examples
        --------
        >>> # Basic usage - each subsample gets independent fit
        >>> results = adjuster.monte_carlo_fit(n_repeats=50, tests=['ks', 'chi2'])
        >>> print(results.sizes.values)  # Sample sizes tested
        >>> ks_pvalues = results['ks_pvalue']  # Access KS p-values directly
        
        >>> # Use fit constraints (e.g., fix location for Weibull)
        >>> results = adjuster.monte_carlo_fit(
        ...     n_repeats=100, 
        ...     tests=['chi2', 'ks'],
        ...     fit_kwargs={'floc': 0}  # Fix location parameter
        ... )
        
        >>> # Use pre-calculated parameters (bypass fitting entirely)
        >>> known_params = (2.0, 0.0, 1.0)  # shape, loc, scale
        >>> results = adjuster.monte_carlo_fit(
        ...     distribution_params=known_params,
        ...     n_repeats=50,
        ...     tests=['chi2', 'ks']
        ... )
        
        >>> # Select specific size and calculate statistics
        >>> size_200_data = results.sel(sizes=200)
        >>> param_medians = results['param_0'].median(dim='repeats')
        
        >>> # Check stability points
        >>> stability = results.attrs['stability_points']
        >>> print(f"KS test stabilizes at size: {stability['ks_pvalue']['size']}")

        Notes
        -----
        **Fitting Strategy:**
        
        By default, each subsample gets an independent fit to evaluate how parameter
        estimates change with sample size. This is the correct approach for stability
        analysis since parameter estimation quality depends on sample size.
        
        **Parameter Constraints:**
        
        Use `fit_kwargs` to impose constraints during fitting (e.g., `floc=0` for 
        Weibull distributions). These constraints apply to all subsample fits.
        
        **Pre-calculated Parameters:**
        
        Use `distribution_params` only when you want to evaluate goodness-of-fit
        with known/fixed parameters across all sample sizes, bypassing fitting entirely.
        
        **Stability Detection:**
        
        The method detects stability by looking for sample sizes where:
        
        1. Test statistics (p-values) become stable across sizes
        2. Parameter estimates converge to consistent values
        
        A moving window approach identifies the first size where values
        remain stable within a tolerance threshold. Stability points are
        stored in the Dataset attributes for easy access.
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

                    # Run goodness-of-fit tests
                    for test in test_list:
                        try:
                            if test == 'ks':
                                ks_res = adj.goodness_of_fit('ks')
                                test_arrays['ks_statistic'][i, rep_j] = ks_res.get('statistic', ks_res.get('ks_statistic', np.nan))
                                test_arrays['ks_pvalue'][i, rep_j] = ks_res.get('p_value', np.nan)
                            elif test == 'chi2':
                                chi_res = adj.goodness_of_fit('chi2', bins=bins, warn_on_normalization=False)
                                test_arrays['chi2_statistic'][i, rep_j] = chi_res.get('statistic', chi_res.get('chi2_statistic', np.nan))
                                test_arrays['chi2_pvalue'][i, rep_j] = chi_res.get('p_value', np.nan)
                            elif test == 'rmse':
                                rmse_res = adj.goodness_of_fit('rmse')
                                test_arrays['rmse'][i, rep_j] = rmse_res.get('rmse', np.nan)
                        except Exception:
                            # Values remain NaN for failed tests
                            pass

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

        # Detect stability points
        stability_points = self._detect_stability_points(data_vars, sizes)

        # Optionally create & save figure
        figure_path = None
        if fig_output_path is not None:
            try:
                fig = self._create_monte_carlo_figure(data_vars, sizes, plot_type, stability_points)
                if fig is not None:
                    fig.savefig(fig_output_path, dpi=150, bbox_inches='tight')
                    figure_path = fig_output_path
            except Exception:
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
                'stability_points': stability_points,
                'figure_path': figure_path,
                'created_by': 'MagicAdjuster.monte_carlo_fit'
            }
        )

        return ds

    def _detect_stability_points(self, data_vars: Dict[str, np.ndarray], sizes: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Detect stability points for each variable using coefficient of variation.
        
        A variable is considered stable when its coefficient of variation
        (std/mean) across repeats becomes consistently low.
        """
        stability_points = {}
        window_size = max(2, len(sizes) // 4)  # Use 25% of sizes as window
        cv_threshold = 0.1  # 10% coefficient of variation threshold
        
        for var_name, data_array in data_vars.items():
            if np.all(np.isnan(data_array)):
                continue
                
            # Calculate coefficient of variation for each size
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
            
            # Find first point where CV stays below threshold for window_size consecutive points
            stable_idx = None
            for i in range(len(cv_values) - window_size + 1):
                window_cvs = cv_values[i:i + window_size]
                if all(cv < cv_threshold for cv in window_cvs if not np.isinf(cv)):
                    stable_idx = i
                    break
            
            if stable_idx is not None:
                stability_points[var_name] = {
                    'size': int(sizes[stable_idx]),
                    'index': stable_idx,
                    'cv_at_stability': cv_values[stable_idx]
                }
            else:
                stability_points[var_name] = {
                    'size': None,
                    'index': None,
                    'cv_at_stability': None
                }
        
        return stability_points

    def _create_monte_carlo_figure(
        self,
        data_vars: Dict[str, np.ndarray],
        sizes: np.ndarray,
        plot_type: str,
        stability_points: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Create 2x3 summary figure (row 1: parameters, row 2: test p-values/statistics).

        Draws a vertical dashed red line at the stability sample size for each
        variable if available in stability_points.
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
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
        axes_flat = axes.ravel()

        def _panel(ax, data: np.ndarray, title: str, var_name: str):
            if plot_type == 'boxplots':
                box_data = [data[i, ~np.isnan(data[i, :])] for i in range(len(sizes))]
                if len(sizes) > 1:
                    width = max(1, (sizes[1] - sizes[0]) * 0.6)
                else:
                    width = 5
                ax.boxplot(box_data, positions=sizes, widths=width)
            else:  # series
                med = np.nanmedian(data, axis=1)
                q25 = np.nanpercentile(data, 25, axis=1)
                q75 = np.nanpercentile(data, 75, axis=1)
                ax.plot(sizes, med, 'o-', lw=1.4, label='Median')
                ax.fill_between(sizes, q25, q75, alpha=0.25, label='IQR')
                ax.legend(frameon=False, fontsize=8)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            # Stability vertical line if exists
            if stability_points:
                sp = stability_points.get(var_name)
                if sp and sp.get('size') is not None:
                    try:
                        ax.axvline(sp['size'], color='red', linestyle='--', linewidth=1, alpha=0.85)
                    except Exception:
                        pass

        # Parameter panels (row 0)
        for col, pname in enumerate(param_names):
            ax = axes[0, col]
            _panel(ax, data_vars[pname], pname.replace('_', ' ').title(), pname)
            ax.set_ylabel('Value')
        # Hide unused param axes in row 0
        for col in range(len(param_names), 3):
            axes[0, col].axis('off')

        # Test panels (row 1)
        for col, tname in enumerate(test_names):
            ax = axes[1, col]
            label = tname.replace('_', ' ').title()
            _panel(ax, data_vars[tname], label, tname)
            ax.set_xlabel('Sample Size')
            if col == 0:
                ax.set_ylabel('Value')
        for col in range(len(test_names), 3):
            axes[1, col].axis('off')

        # Shared formatting
        for ax in axes_flat:
            if ax.has_data():
                ax.tick_params(axis='x', rotation=0)

        fig.suptitle('Monte Carlo Stability Summary', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig

    def _make_boxplots_figure(
        self,
        sizes: np.ndarray,
        summary: Dict[str, Any],
        param_values_per_size: Dict[int, List[List[float]]],
        test_list: List[str],
        max_params: int,
    ):
        """
        Build and return a matplotlib Figure with boxplots for the chosen test and
        up to the first three fitted parameters. Returns None on failure or when
        matplotlib is not available.

        Parameters
        ----------
        sizes
            Array-like of sample sizes used as x-axis labels.
        summary
            The summary dict generated by monte_carlo_fit (contains 'tests').
        param_values_per_size
            Mapping param_index -> list-of-lists of values per sample size.
        test_list
            List of test names (e.g. ['chi2', 'ks']). The first entry is used for
            the primary boxplot.
        max_params
            Maximum number of parameters discovered during fitting.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        try:
            n_param_plots = min(3, max_params)

            # main figure: p-values / metric boxplots
            fig_main, ax_main = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
            bp_data = summary['tests'][test_list[0]]['values_per_size']
            ax_main.boxplot(bp_data, labels=[str(s) for s in sizes], showfliers=False)
            ax_main.set_title(f"CPS boxplot for {test_list[0]}")
            ax_main.set_xlabel('sample size')
            ax_main.set_ylabel('p-value' if test_list[0] in ['chi2', 'ks'] else test_list[0])

            fig_params = None
            if n_param_plots > 0:
                fig_params, axes = plt.subplots(n_param_plots, 1, figsize=(10, 3 * n_param_plots), constrained_layout=True)
                if n_param_plots == 1:
                    axes = [axes]
                for pi in range(n_param_plots):
                    ax = axes[pi]
                    pdata = param_values_per_size.get(pi, [])
                    ax.boxplot(pdata, labels=[str(s) for s in sizes], showfliers=False)
                    ax.set_title(f'Parameter {pi} distribution across sample sizes')
                    ax.set_xlabel('sample size')
                    ax.set_ylabel(f'param_{pi}')

            return fig_main, fig_params
        except Exception:
            return None, None

    def _make_series_figure(
        self,
        sizes: np.ndarray,
        summary: Dict[str, Any],
        param_values_per_size: Dict[int, List[List[float]]],
        test_list: List[str],
        max_params: int,
    ):
        """
        Build and return a matplotlib Figure showing a series (line) for the chosen
        test medians across sample sizes, with shaded area showing dispersion (e.g.
        25-75 percentile). Also plots medians for up to three parameters with shaded
        dispersion. Returns None on failure or when matplotlib is not available.
        """
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

            # Prepare test median and quartiles
            vals = summary['tests'][test_list[0]]['values_per_size']
            medians = [np.median(v) if len(v) > 0 else np.nan for v in vals]
            q1 = [np.percentile(v, 25) if len(v) > 0 else np.nan for v in vals]
            q3 = [np.percentile(v, 75) if len(v) > 0 else np.nan for v in vals]

            x = np.arange(len(sizes))
            ax.plot(x, medians, marker='o', label=f'{test_list[0]} median')
            ax.fill_between(x, q1, q3, alpha=0.3, label=f'{test_list[0]} 25-75%')
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in sizes], rotation=45)
            ax.set_xlabel('sample size')
            ax.set_ylabel('p-value' if test_list[0] in ['chi2', 'ks'] else test_list[0])
            ax.set_title(f'Stability series for {test_list[0]}')
            ax.legend()

            # parameter series (up to 3)
            n_param_plots = min(3, max_params)
            fig_params = None
            if n_param_plots > 0:
                fig_params, axes = plt.subplots(n_param_plots, 1, figsize=(10, 3 * n_param_plots), constrained_layout=True)
                if n_param_plots == 1:
                    axes = [axes]
                for pi in range(n_param_plots):
                    pdata = param_values_per_size.get(pi, [])
                    meds = [np.median(v) if len(v) > 0 else np.nan for v in pdata]
                    q1p = [np.percentile(v, 25) if len(v) > 0 else np.nan for v in pdata]
                    q3p = [np.percentile(v, 75) if len(v) > 0 else np.nan for v in pdata]
                    axp = axes[pi]
                    axp.plot(x, meds, marker='o', label=f'param_{pi} median')
                    axp.fill_between(x, q1p, q3p, alpha=0.25, label='25-75%')
                    axp.set_xticks(x)
                    axp.set_xticklabels([str(s) for s in sizes], rotation=45)
                    axp.set_xlabel('sample size')
                    axp.set_ylabel(f'param_{pi}')
                    axp.legend()

            return fig, fig_params
        except Exception:
            return None

    def _aggregate_and_detect_stability(
        self,
        results: Dict[str, Any],
        sizes: np.ndarray,
        test_list: List[str],
        max_params: int,
    ) -> Tuple[Dict[str, Any], Dict[int, List[List[float]]], Dict[int, List[float]]]:
        """
        Aggregate monte carlo `results` and detect stability (CPS) for tests and parameters.

        Returns (summary, param_values_per_size, param_medians)
        """
        summary: Dict[str, Any] = {'sizes': sizes.tolist(), 'tests': {}, 'params': {}}

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
            tol = 0.01
            window = 3
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
        


## TO-DO:
    def _detect_stability(self, results: Dict[str, Any], sizes: np.ndarray) -> Dict[str, Any]:
        """Detect stability in Monte Carlo results using moving window approach."""
        # Simple implementation - can be enhanced later
        return {
            'stable_size': None,
            'window_size': 3,
            'tolerance': 0.1,
            'is_stable': False
        }
    
    def _create_stability_plot(self, results: Dict[str, Any]) -> tuple:
        """Create stability plots for Monte Carlo results."""
        # Simple implementation returning None figures - can be enhanced later
        return (None, None)


# When set a new distribution to an already used variable, it will override the distribution form previous variable
# For example:
# fitted_data_weibull = data.fit_distribution('weibull')
# fitted_data_norm = data.fit_distribution('norm')
#
# The distribution from "fitted_data_weibull" will return the norm dist.

 