"""
Statistical distribution fitting and adjustment for wind data
"""

from matplotlib.pylab import seed
import numpy as np
from scipy import stats
from sklearn.metrics import root_mean_squared_error
from typing import Union, Dict, Any, Optional, Tuple, List
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

    def goodness_of_fit(self, method: str, bins: Union[str, int] = 'doane'):
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
            if discrepancy > 1e-6:
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
            sample_sizes: Optional[List[int]] = None,
            n_subsamples: int = 20,
            n_repeats: Optional[int] = None,
            min_size: int = 100,
            sampling: str = 'disjoint',  # 'random'|'disjoint'|'bootstrap'
            seed: Optional[int] = None,
            distribution: Optional[Union[str, object]] = None,
            tests: Optional[List[str]] = None,  # e.g. ['chi2','ks','rmse'] or None -> ['chi2']
            bins: Union[str, int] = 'doane',
            n_jobs: int = 1,
            store_raw: bool = False,
            fit_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
        """
        Monte Carlo orchestration (index generation stage).

        Behavior:
        - If `sample_sizes` is provided (list of ints), those sizes are used.
        - If `sample_sizes` is None, a grid of `n_subsamples` sizes is generated
          between `min_size` and the full sample size.

        Parameters
        ----------
        sample_sizes : list[int], optional
            Explicit list of subsample sizes to evaluate (e.g. [100, 145, 190]).
            If None, sizes will be generated using `n_subsamples` and `min_size`.
        n_subsamples : int
            When `sample_sizes` is None, number of distinct sizes to generate.
        n_repeats : int, optional
            Number of subsamples (repeats) to generate per size. If None a sensible
            default is chosen based on the smallest size.
        min_size : int
            Minimum subsample size used when building the grid.
        sampling : {'random','bootstrap','disjoint'}
            Sampling strategy passed to `_generate_subsample_indices`.
        seed : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        dict
            Results dict with generated sizes and index arrays per size:
            {
                'sizes': np.ndarray,
                'indices': { size_int: ndarray(shape=(n_repeats, size)), ... },
                'meta': {...}
            }
        """
        N = len(self.data)
        if N == 0:
            raise ValueError("No data available for monte_carlo_fit.")

        # Determine sizes
        if sample_sizes is None:
            min_size = max(1, int(min_size))
            sizes = np.unique(np.linspace(min_size, N, int(n_subsamples), dtype=int))
            sizes = sizes[sizes > 0]
            if sizes.size == 0:
                raise ValueError("Generated empty sizes grid; check min_size and n_subsamples values.")
        else:
            sizes = np.unique(np.asarray(sample_sizes, dtype=int))
            if sizes.size == 0:
                raise ValueError("`sample_sizes` provided is empty or invalid")

        # Determine repeats per size
        if n_repeats is None:
            smallest = int(sizes[0])
            n_repeats = max(10, min(100, N // smallest))

        results: Dict[str, Any] = {
            'sizes': sizes,
            'indices': {},
            'results': {},  # per-size -> list of per-repeat dicts with params and gof
            'meta': {
                'n_sizes': int(sizes.size),
                'n_repeats': int(n_repeats),
                'n_subsamples_param': int(n_subsamples),
                'sampling': sampling,
                'seed': seed,
                'min_size': int(min_size)
            }
        }

        if seed is None:
            child_seq = [None] * sizes.size
        else:
            ss = np.random.SeedSequence(seed)
            child_seq = ss.spawn(sizes.size)  # list of SeedSequence objects

        for i, size in enumerate(sizes):
            child_seed = child_seq[i]  # either SeedSequence or None

            idx_list = self._generate_subsample_indices(
                size=int(size),
                n_repeats=int(n_repeats),
                sampling=sampling,
                seed=child_seed
            )

            try:
                idx_arr = np.asarray(idx_list, dtype=int)
            except Exception:
                idx_arr = idx_list

            results['indices'][int(size)] = idx_arr

            # Now run fits + goodness-of-fit for each repeat
            per_size_results = []
            # determine which tests to run
            test_list = [t.lower() for t in tests] if tests is not None else ['chi2']

            for rep_i in range(len(idx_list)):
                rep_result: Dict[str, Any] = {'params': None, 'gof': {}, 'error': None}
                try:
                    sub_idx = idx_list[rep_i]
                    subdata = self.data[sub_idx]

                    # create temporary DataProcessor and fit
                    dp = DataProcessor(subdata)
                    adj = dp._get_adjuster()

                    # choose distribution: explicit param > existing fitted in self
                    dist_to_fit = distribution if distribution is not None else self.fitted_distribution
                    if dist_to_fit is None:
                        raise ValueError("No distribution provided and no fitted distribution available on self.")

                    # fit
                    adj.fit_distribution(dist_to_fit, **(fit_kwargs or {}))
                    params = adj.get_fitted_params()
                    rep_result['params'] = params

                    # run requested goodness-of-fit tests
                    if 'chi2' in test_list:
                        try:
                            chi_res = adj.goodness_of_fit('chi2', bins=bins)
                            rep_result['gof']['chi2'] = chi_res
                        except Exception as e:
                            rep_result['gof']['chi2'] = {'error': str(e)}

                    if 'ks' in test_list:
                        try:
                            ks_res = adj.goodness_of_fit('ks')
                            rep_result['gof']['ks'] = ks_res
                        except Exception as e:
                            rep_result['gof']['ks'] = {'error': str(e)}

                    if 'rmse' in test_list:
                        try:
                            rmse_res = adj.goodness_of_fit('rmse', bins=bins)
                            rep_result['gof']['rmse'] = rmse_res
                        except Exception as e:
                            rep_result['gof']['rmse'] = {'error': str(e)}

                except Exception as e:
                    rep_result['error'] = str(e)

                per_size_results.append(rep_result)

            results['results'][int(size)] = per_size_results

        # --- Aggregation and CPS (stability) detection ---
        # Which tests to summarize
        test_list = [t.lower() for t in tests] if tests is not None else ['chi2']

        summary: Dict[str, Any] = {'sizes': sizes.tolist(), 'tests': {}, 'params': {}}

        # Collect parameter values per size
        # determine max number of params returned by fits
        max_params = 0
        for size in sizes:
            per = results['results'].get(int(size), [])
            for rep in per:
                if rep.get('params') is None:
                    continue
                max_params = max(max_params, len(rep['params']))

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
                            # try statistic value fallback
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
                        inflection_idx = j + (window - 1)  # choose end of stable window
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
            # for each param index, collect values across repeats
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

        results['summary'] = summary

        # --- Create figure: boxplots for tests and first up to 3 parameters ---
        try:
            import matplotlib.pyplot as plt

            n_param_plots = min(3, max_params)
            n_test_plots = len(test_list)
            total_rows = 1 + n_param_plots
            fig, axes = plt.subplots(total_rows, 1, figsize=(8, 3 * total_rows), constrained_layout=True)
            if total_rows == 1:
                axes = [axes]

            # p-values / metric boxplots
            p_axes = axes[0]
            bp_data = [summary['tests'][t]['values_per_size'] for t in [test_list[0]]][0]
            # boxplot expects list of sequences per x position
            p_axes.boxplot(bp_data, labels=[str(s) for s in sizes], showfliers=False)
            p_axes.set_title(f"CPS boxplot for {test_list[0]}")
            p_axes.set_xlabel('sample size')
            p_axes.set_ylabel('p-value' if test_list[0] in ['chi2', 'ks'] else test_list[0])

            # parameter boxplots for first up to 3 params
            for pi in range(n_param_plots):
                ax = axes[1 + pi]
                pdata = param_values_per_size.get(pi, [])
                ax.boxplot(pdata, labels=[str(s) for s in sizes], showfliers=False)
                ax.set_title(f'Parameter {pi} distribution across sample sizes')
                ax.set_xlabel('sample size')
                ax.set_ylabel(f'param_{pi}')

            results['figure'] = fig
        except Exception:
            # figure generation is optional; ignore errors and continue
            results['figure'] = None

        return results


## TO-DO:
# When set a new distribution to an already used variable, it will override the distribution form previous variable
# For example:
# fitted_data_weibull = data.fit_distribution('weibull')
# fitted_data_norm = data.fit_distribution('norm')
#
# The distribution from "fitted_data_weibull" will return the norm dist.

 