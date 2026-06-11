"""
Statistical distribution fitting and adjustment for wind data
"""

import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from typing import Union, Dict, Any, Optional, Tuple, List
import warnings
from dataclasses import dataclass
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
        # 'dpareto_lognorm': stats.dpareto_lognorm,  # Not available in SciPy 1.14
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
        # 'landau': stats.landau,  # Not available in SciPy 1.14
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


# ---------------------------------------------------------------------------
# Goodness-of-fit helpers (module-level so both FitResult and MagicAdjuster
# can share them without circular dependency)
# ---------------------------------------------------------------------------

def _num_bins(data: np.ndarray, bins: Union[str, int]) -> int:
    """Compute number of histogram bins for a dataset."""
    N = len(data)
    if isinstance(bins, int):
        return bins
    b = bins.lower()
    if b == 'sturges':
        return int(1 + np.log2(N))
    if b == 'rice':
        return int(2 * N ** (1 / 3))
    if b == 'scott':
        bin_width = 3.5 * np.std(data) / N ** (1 / 3)
        return max(1, int((data.max() - data.min()) / bin_width))
    if b == 'freedman-diaconis':
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / N ** (1 / 3)
        return max(1, int((data.max() - data.min()) / bin_width))
    if b == 'doane':
        g1 = stats.skew(data)
        sigma_g1 = np.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
        return int(1 + np.log2(N) + np.log2(1 + abs(g1) / sigma_g1))
    raise ValueError(f"Unknown binning method: {bins!r}. "
                     "Use 'sturges', 'rice', 'scott', 'freedman-diaconis', 'doane', or an int.")


def _gof(data: np.ndarray, distribution, params: tuple, method: str,
         bins: Union[str, int] = 'doane', warn_on_normalization: bool = True):
    """
    Compute goodness-of-fit metric for *data* against a frozen distribution.

    Parameters
    ----------
    data : ndarray
    distribution : scipy continuous distribution (unfrozen)
    params : tuple – fitted parameters
    method : 'chi2', 'ks', 'rmse', 'aic', 'bic'
    bins : binning spec for chi2/rmse
    warn_on_normalization : bool

    Returns
    -------
    dict or float
    """
    if method.lower() in ('chisquared', 'chi2'):
        n_bins = _num_bins(data, bins)
        observed_freq, bin_edges = np.histogram(data, bins=n_bins)
        bin_probs = np.diff(distribution.cdf(bin_edges, *params))
        expected_freq = bin_probs * len(data)
        discrepancy = abs(expected_freq.sum() - observed_freq.sum())
        if discrepancy > 1e-6 and warn_on_normalization:
            warnings.warn(
                f"Normalizing expected frequencies. Original sum: {expected_freq.sum():.6f}, "
                f"Target sum: {observed_freq.sum()}"
            )
        if discrepancy > 1e-10:
            expected_freq *= observed_freq.sum() / expected_freq.sum()
        chi_stats = stats.chisquare(observed_freq, f_exp=expected_freq)
        return {
            'chi2_statistic': chi_stats.statistic,
            'p_value': chi_stats.pvalue,
            'n_bins': n_bins,
            'observed_freq': observed_freq,
            'expected_freq': expected_freq,
        }

    if method.lower() in ('kolmogorov-smirnov', 'ks'):
        ks_stats = stats.kstest(data, distribution.cdf, args=params)
        return {'ks_statistic': ks_stats.statistic, 'p_value': ks_stats.pvalue}

    if method.lower() in ('root-mean-square-error', 'rmse'):
        sorted_data = np.sort(data)
        empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        theoretical_cdf = distribution.cdf(sorted_data, *params)
        return float(np.sqrt(np.mean((empirical_cdf - theoretical_cdf) ** 2)))

    if method.lower() == 'aic':
        log_likelihood = np.sum(distribution.logpdf(data, *params))
        k = len(params)
        return float(2 * k - 2 * log_likelihood)

    if method.lower() == 'bic':
        log_likelihood = np.sum(distribution.logpdf(data, *params))
        k = len(params)
        n = len(data)
        return float(k * np.log(n) - 2 * log_likelihood)

    raise ValueError(f"Unknown goodness-of-fit method: {method!r}. "
                     "Available: 'chi2', 'ks', 'rmse', 'aic', 'bic'")


# ---------------------------------------------------------------------------
# FitResult – immutable result of a single distribution fit
# ---------------------------------------------------------------------------

@dataclass(eq=False, frozen=True)
class FitResult:
    """
    Immutable result of fitting a distribution to data.

    Attributes
    ----------
    distribution : scipy continuous distribution (unfrozen)
        The distribution family (e.g., ``scipy.stats.weibull_min``).
    name : str
        Human-readable distribution name.
    params : tuple of float
        Fitted parameters as returned by ``distribution.fit()``.
    data : ndarray
        **Shared reference** to the array that was fitted.  No copy is made;
        the cost of each ``FitResult`` is just the metadata tuple + scalars.

    Examples
    --------
    >>> import magica as ma, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.weibull(2, 500) * 8
    >>> processor = ma.read_data(data)
    >>> fit = processor.fit('weibull', floc=0)
    >>> fit.pdf()          # evaluate at the original data
    >>> fit.cdf(np.array([5, 10, 15]))  # evaluate at custom points
    >>> fit.ppf(0.99)      # 99th-percentile quantile
    >>> fit.goodness_of_fit('ks')
    """

    distribution: object
    name: str
    params: tuple
    data: np.ndarray

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def frozen(self):
        """Return a frozen (parameterized) scipy distribution."""
        return self.distribution(*self.params)

    @property
    def info(self) -> Dict[str, Any]:
        """Summary dict: name, parameters, num_params, data_size."""
        return {
            'name': self.name,
            'parameters': self.params,
            'num_params': len(self.params),
            'data_size': len(self.data),
        }

    # ------------------------------------------------------------------
    # Distribution methods with smart defaults (data used when x is None)
    # ------------------------------------------------------------------

    def pdf(self, x=None) -> np.ndarray:
        """Probability density; defaults to evaluating at the fitted data."""
        if x is None:
            x = self.data
        return self.distribution.pdf(x, *self.params)

    def cdf(self, x=None) -> np.ndarray:
        """Cumulative distribution function; defaults to fitted data."""
        if x is None:
            x = self.data
        return self.distribution.cdf(x, *self.params)

    def sf(self, x=None) -> np.ndarray:
        """Survival function (1 - CDF); defaults to fitted data."""
        if x is None:
            x = self.data
        return self.distribution.sf(x, *self.params)

    def logpdf(self, x=None) -> np.ndarray:
        """Log probability density; defaults to fitted data."""
        if x is None:
            x = self.data
        return self.distribution.logpdf(x, *self.params)

    def logcdf(self, x=None) -> np.ndarray:
        """Log cumulative distribution; defaults to fitted data."""
        if x is None:
            x = self.data
        return self.distribution.logcdf(x, *self.params)

    def logsf(self, x=None) -> np.ndarray:
        """Log survival function; defaults to fitted data."""
        if x is None:
            x = self.data
        return self.distribution.logsf(x, *self.params)

    def ppf(self, q) -> np.ndarray:
        """Percent-point function (inverse CDF); *q* is required."""
        return self.distribution.ppf(q, *self.params)

    def isf(self, q) -> np.ndarray:
        """Inverse survival function; *q* is required."""
        return self.distribution.isf(q, *self.params)

    def rvs(self, size=None, random_state=None) -> np.ndarray:
        """Random variates from the fitted distribution."""
        return self.distribution.rvs(*self.params, size=size, random_state=random_state)

    def stats(self, moments='mv'):
        """Mean, variance (and skewness/kurtosis) of the fitted distribution."""
        return self.distribution.stats(*self.params, moments=moments)

    # ------------------------------------------------------------------
    # Goodness-of-fit
    # ------------------------------------------------------------------

    def goodness_of_fit(
        self,
        method: str,
        bins: Union[str, int] = 'doane',
        warn_on_normalization: bool = True,
    ):
        """
        Goodness-of-fit test for this fit against its own data.

        Parameters
        ----------
        method : str
            ``'chi2'``, ``'ks'``, ``'rmse'``, ``'aic'``, or ``'bic'``.
        bins : int or str, default ``'doane'``
            Binning method for ``'chi2'`` and ``'rmse'`` tests.
        warn_on_normalization : bool, default True

        Returns
        -------
        dict or float
            Dict for ``'chi2'`` and ``'ks'``; float for ``'rmse'``, ``'aic'``, ``'bic'``.
        """
        return _gof(self.data, self.distribution, self.params,
                    method, bins, warn_on_normalization)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        param_str = ", ".join(f"{p:.4g}" for p in self.params)
        return (f"FitResult(distribution='{self.name}', "
                f"params=({param_str}), data_size={len(self.data)})")


# ---------------------------------------------------------------------------
# MagicAdjuster
# ---------------------------------------------------------------------------

class MagicAdjuster:
    """
    Stateless fitter: wraps a data array and fits distributions, returning
    immutable :class:`FitResult` objects.

    Internally retains ``fitted_distribution`` and ``distribution_name`` only
    to enable :meth:`monte_carlo_fit` to know which distribution to resample,
    without requiring the caller to pass it explicitly.

    Examples
    --------
    >>> import magica as ma, numpy as np
    >>> data = ma.read_data(np.random.weibull(2, 500) * 8)
    >>> adjuster = data._get_adjuster()
    >>> fit = adjuster.fit_distribution('weibull', floc=0)
    >>> fit.cdf(np.array([5., 10.]))
    >>> mc = adjuster.monte_carlo_fit(tests=['ks', 'rmse'])
    """

    def __init__(self, data_processor: DataProcessor):
        if data_processor.data is None:
            raise ValueError("DataProcessor must have data loaded.")

        self.data_processor = data_processor
        self.data = data_processor.data  # shared reference, no copy

        # Minimal internal state – only used to remember which distribution
        # to resample in monte_carlo_fit.
        self.fitted_distribution = None
        self.distribution_name = None

    # ------------------------------------------------------------------
    # Core fitting
    # ------------------------------------------------------------------

    def fit_distribution(self, distribution: Union[str, object], **kwargs) -> FitResult:
        """
        Fit a distribution to the data and return an immutable :class:`FitResult`.

        Each call is independent: no state from a previous call is carried over,
        and two simultaneous fits coexist without interference.

        Parameters
        ----------
        distribution : str or scipy continuous distribution
            ``'weibull'``, ``'gamma'``, ``stats.weibull_min``, etc.
        **kwargs
            Passed verbatim to ``distribution.fit()`` (e.g., ``floc=0``).

        Returns
        -------
        FitResult
            Immutable result containing the distribution, fitted parameters,
            and a **shared reference** to the data array (no copy).

        Examples
        --------
        >>> fit_w = adjuster.fit_distribution('weibull', floc=0)
        >>> fit_g = adjuster.fit_distribution('gamma')
        >>> # Both coexist independently:
        >>> fit_w.name, fit_g.name
        ('weibull', 'gamma')
        """
        if isinstance(distribution, str):
            distribution_map = get_available_distributions()
            name = distribution.lower()
            if name not in distribution_map:
                raise ValueError(f"Unknown distribution {distribution!r}. "
                                 f"Available: {sorted(distribution_map.keys())}")
            dist_obj = distribution_map[name]
        else:
            dist_obj = distribution
            name = getattr(distribution, 'name', str(distribution))

        try:
            params = dist_obj.fit(self.data, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to fit {name!r}: {e}") from e

        # Remember for monte_carlo_fit (minimal state, not a fit result)
        self.fitted_distribution = dist_obj
        self.distribution_name = name

        return FitResult(
            distribution=dist_obj,
            name=name,
            params=tuple(params),
            data=self.data,  # shared reference
        )

    # ------------------------------------------------------------------
    # Goodness-of-fit (kept for backward-compat / internal use in MC loop)
    # ------------------------------------------------------------------

    def goodness_of_fit(self, method: str, bins: Union[str, int] = 'doane',
                        warn_on_normalization: bool = True):
        """
        Goodness-of-fit against the last fitted distribution.

        Prefer calling ``fit_result.goodness_of_fit()`` instead, which is
        self-contained and explicit.
        """
        if self.fitted_distribution is None:
            raise ValueError("No distribution fitted yet. Call fit_distribution() first.")
        # Reconstruct params by refitting? No – for internal MC use we always
        # call fit_distribution first, so fitted_distribution / params are current.
        # But we no longer store fitted_params on self after the refactor...
        # We raise a clear error to guide migration.
        raise AttributeError(
            "MagicAdjuster.goodness_of_fit() is deprecated. "
            "Use the returned FitResult: fit = adjuster.fit_distribution(...); "
            "fit.goodness_of_fit(method)."
        )

    # ------------------------------------------------------------------
    # Histogram binning helpers (kept for backward-compat / MC internals)
    # ------------------------------------------------------------------

    def get_bin_number_sturges(self) -> int:
        N = len(self.data)
        return int(1 + np.log2(N))

    def get_bin_number_rice(self) -> int:
        N = len(self.data)
        return int(2 * N ** (1 / 3))

    def get_bin_number_freedman_diaconis(self) -> int:
        iqr = np.percentile(self.data, 75) - np.percentile(self.data, 25)
        bin_width = 2 * iqr / len(self.data) ** (1 / 3)
        return max(1, int((max(self.data) - min(self.data)) / bin_width))

    def get_bin_number_scott(self) -> int:
        bin_width = 3.5 * np.std(self.data) / len(self.data) ** (1 / 3)
        return max(1, int((max(self.data) - min(self.data)) / bin_width))

    def get_bin_number_doane(self) -> int:
        N = len(self.data)
        g1 = stats.skew(self.data)
        sigma_g1 = np.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
        return int(1 + np.log2(N) + np.log2(1 + abs(g1) / sigma_g1))

    def get_num_bins(self, bins='doane') -> int:
        return _num_bins(self.data, bins)

    # ------------------------------------------------------------------
    # Subsampling helpers for Monte Carlo
    # ------------------------------------------------------------------

    def _generate_subsample_indices(
        self,
        size: int,
        n_repeats: int,
        sampling: str = 'random',
        seed: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Generate index arrays for subsamples.

        Parameters
        ----------
        size : int
        n_repeats : int
        sampling : {'random','bootstrap','disjoint'}
        seed : int, optional

        Returns
        -------
        list of ndarray
        """
        rng = np.random.default_rng(seed)
        N = len(self.data)

        if sampling not in {'random', 'bootstrap', 'disjoint'}:
            raise ValueError(f"Unknown sampling strategy: {sampling!r}")
        if sampling == 'random' and size > N:
            raise ValueError("For 'random' sampling size must be <= len(data). "
                             "Use 'bootstrap' to allow size > N.")

        indices: List[np.ndarray] = []

        if sampling == 'bootstrap':
            for _ in range(n_repeats):
                indices.append(rng.integers(0, N, size=size))
        elif sampling == 'random':
            for _ in range(n_repeats):
                indices.append(rng.choice(N, size=size, replace=False))
        else:  # disjoint
            if size <= 0:
                raise ValueError("size must be > 0 for disjoint sampling")
            per_pass = N // size
            if per_pass == 0:
                raise ValueError("disjoint sampling not possible: size > len(data)")
            needed = n_repeats
            while needed > 0:
                perm = rng.permutation(N)
                for i in range(per_pass):
                    if needed == 0:
                        break
                    indices.append(perm[i * size:(i + 1) * size])
                    needed -= 1

        return indices

    # ------------------------------------------------------------------
    # Monte Carlo stability analysis
    # ------------------------------------------------------------------

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
    ):
        """
        Monte Carlo stability analysis for distribution fitting.

        Evaluates how stable distribution parameters and goodness-of-fit
        statistics are across different sample sizes, helping to identify the
        minimum sample size required for reliable parameter estimation.

        Parameters
        ----------
        sizes : list of int, optional
            Explicit list of sample sizes to test. If None, a grid between
            ``min_size`` and ``max_size`` is created.
        n_repeats : int, default 20
            Repetitions per size.
        tests : list of str, default ``['ks']``
            Goodness-of-fit tests: ``'ks'``, ``'chi2'``, ``'rmse'``.
            Include ``'rmse'`` for best stability detection.
        stability_method : str, default ``'kneedle'``
            ``'cv'``, ``'kneedle'``, ``'plateau'``, ``'aggregate'``,
            ``'detect'`` (alias for ``'cv'``), or ``None``/``'none'``.
        fig_output_path : str, optional
            If given, save a 2×3 summary figure to this path.
        plot_type : str, default ``'series'``
            Panel style: ``'series'`` or ``'boxplots'``.
        sampling : str, default ``'random'``
            ``'random'``, ``'bootstrap'``, or ``'disjoint'``.
        seed : int, optional
            Random seed for reproducibility.
        min_size : int, default 50
            Minimum size for auto-generated grid.
        max_size : int, optional
            Maximum size (defaults to full dataset).
        n_sizes : int, default 10
            Grid points when auto-generating sizes.
        distribution_params : tuple, optional
            Fixed parameters to bypass fitting (GOF-only mode).
        **kwargs
            Extra parameters for binning (``bins``), fitting constraints
            (``fit_kwargs``), and stability detection tuning.

        Returns
        -------
        xarray.Dataset
            Monte Carlo results with ``sizes`` and ``repeats`` dimensions.

        Examples
        --------
        >>> fit = data.fit('weibull', floc=0)
        >>> adjuster = data._get_adjuster()
        >>> results = adjuster.monte_carlo_fit(
        ...     tests=['ks', 'rmse'], stability_method='kneedle'
        ... )
        >>> print(results.attrs['recommended_size'])
        """
        if self.fitted_distribution is None and distribution_params is None:
            raise ValueError(
                "Call fit_distribution() before monte_carlo_fit(), "
                "or pass distribution_params for GOF-only mode."
            )

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
                raise ValueError("Generated empty sizes grid; check min_size and n_sizes.")
        else:
            sizes = np.unique(np.asarray(sizes, dtype=int))
            if sizes.size == 0:
                raise ValueError("`sizes` is empty or invalid.")

        if n_repeats is None:
            n_repeats = max(10, min(100, N // int(sizes[0])))

        test_list = [t.lower() for t in tests] if tests else ['ks']

        # Determine parameter count
        if distribution_params is not None:
            max_params = len(distribution_params)
        else:
            sample_idx = np.random.choice(len(self.data), min(100, len(self.data)), replace=False)
            probe_fit = FitResult(
                distribution=self.fitted_distribution,
                name=self.distribution_name,
                params=tuple(self.fitted_distribution.fit(self.data[sample_idx], **fit_kwargs)),
                data=self.data[sample_idx],
            )
            max_params = len(probe_fit.params)

        # Initialise storage
        n_sizes = len(sizes)
        param_arrays = {f'param_{i}': np.full((n_sizes, n_repeats), np.nan)
                        for i in range(max_params)}
        test_arrays: Dict[str, np.ndarray] = {}
        for test in test_list:
            if test == 'ks':
                test_arrays['ks_statistic'] = np.full((n_sizes, n_repeats), np.nan)
                test_arrays['ks_pvalue'] = np.full((n_sizes, n_repeats), np.nan)
            elif test == 'chi2':
                test_arrays['chi2_statistic'] = np.full((n_sizes, n_repeats), np.nan)
                test_arrays['chi2_pvalue'] = np.full((n_sizes, n_repeats), np.nan)
            elif test == 'rmse':
                test_arrays['rmse'] = np.full((n_sizes, n_repeats), np.nan)

        if seed is None:
            child_seq = [None] * n_sizes
        else:
            ss = np.random.SeedSequence(seed)
            child_seq = ss.spawn(n_sizes)

        aggregate_results = (
            {'results': {int(s): [] for s in sizes}}
            if stability_method == 'aggregate' else None
        )

        # Main MC loop
        for i, size in tqdm(list(enumerate(sizes)), desc='Monte Carlo sizes'):
            idx_list = self._generate_subsample_indices(
                size=int(size), n_repeats=int(n_repeats),
                sampling=sampling, seed=child_seq[i],
            )
            for rep_j in trange(len(idx_list), desc=f'size={size}', leave=False):
                try:
                    subdata = self.data[idx_list[rep_j]]

                    if distribution_params is not None:
                        params = distribution_params
                        sub_fit = FitResult(
                            distribution=self.fitted_distribution,
                            name=self.distribution_name,
                            params=tuple(params),
                            data=subdata,
                        )
                    else:
                        sub_fit = FitResult(
                            distribution=self.fitted_distribution,
                            name=self.distribution_name,
                            params=tuple(self.fitted_distribution.fit(subdata, **fit_kwargs)),
                            data=subdata,
                        )
                        params = sub_fit.params

                    for pidx, pval in enumerate(params):
                        if pidx < max_params:
                            param_arrays[f'param_{pidx}'][i, rep_j] = pval

                    if aggregate_results is not None:
                        rep_entry: Dict[str, Any] = {'params': params, 'gof': {}}

                    for test in test_list:
                        try:
                            if test == 'ks':
                                ks_res = sub_fit.goodness_of_fit('ks')
                                test_arrays['ks_statistic'][i, rep_j] = ks_res.get(
                                    'statistic', ks_res.get('ks_statistic', np.nan))
                                test_arrays['ks_pvalue'][i, rep_j] = ks_res.get('p_value', np.nan)
                                if aggregate_results is not None:
                                    rep_entry['gof']['ks'] = ks_res
                            elif test == 'chi2':
                                chi_res = sub_fit.goodness_of_fit(
                                    'chi2', bins=bins, warn_on_normalization=False)
                                test_arrays['chi2_statistic'][i, rep_j] = chi_res.get(
                                    'statistic', chi_res.get('chi2_statistic', np.nan))
                                test_arrays['chi2_pvalue'][i, rep_j] = chi_res.get('p_value', np.nan)
                                if aggregate_results is not None:
                                    rep_entry['gof']['chi2'] = chi_res
                            elif test == 'rmse':
                                test_arrays['rmse'][i, rep_j] = sub_fit.goodness_of_fit('rmse')
                                if aggregate_results is not None:
                                    rep_entry['gof']['rmse'] = test_arrays['rmse'][i, rep_j]
                        except Exception:
                            pass

                    if aggregate_results is not None:
                        aggregate_results['results'][int(size)].append(rep_entry)

                except Exception:
                    pass

        # Assemble data vars
        data_vars: Dict[str, np.ndarray] = {}
        data_vars.update(param_arrays)
        data_vars.update(test_arrays)

        coords = {'sizes': sizes, 'repeats': np.arange(n_repeats)}

        # Stability detection
        if stability_method is None or stability_method.lower() == 'none':
            stability_points = {
                k: {'size': None, 'index': None, 'cv_at_stability': None,
                    'smoothed_curve': None, 'method': 'none'}
                for k in list(param_arrays) + list(test_arrays)
            }
        elif stability_method.lower() == 'detect':
            stability_points = self._detect_stability_unified(data_vars, sizes, method='cv', **kwargs)
        elif stability_method.lower() in ('cv', 'kneedle', 'plateau'):
            stability_points = self._detect_stability_unified(
                data_vars, sizes, method=stability_method.lower(), **kwargs)
        elif stability_method.lower() == 'aggregate' and aggregate_results is not None:
            agg_summary, _, _ = self._aggregate_and_detect_stability(
                aggregate_results, sizes, test_list, max_params)
            stability_points = {}
            for test in agg_summary.get('tests', {}):
                inf_idx = agg_summary['tests'][test].get('inflection_index')
                stability_points[test] = {
                    'size': int(sizes[inf_idx]) if inf_idx is not None else None,
                    'index': int(inf_idx) if inf_idx is not None else None,
                    'cv_at_stability': None, 'smoothed_curve': None, 'method': 'aggregate',
                }
            p_inf_idx = agg_summary['params'].get('inflection_index')
            stability_points['param_0'] = {
                'size': int(sizes[p_inf_idx]) if p_inf_idx is not None else None,
                'index': int(p_inf_idx) if p_inf_idx is not None else None,
                'cv_at_stability': None, 'smoothed_curve': None, 'method': 'aggregate',
            }
        else:
            raise ValueError(f"Unknown stability_method: {stability_method!r}. "
                             "Options: 'cv', 'kneedle', 'plateau', 'aggregate', 'detect', 'none'")

        # Recommended size
        recommended_size = None
        primary_metric = None
        for metric in ('rmse', 'ks_pvalue', 'chi2_pvalue', 'param_0'):
            if metric in stability_points and stability_points[metric]['size'] is not None:
                recommended_size = stability_points[metric]['size']
                primary_metric = metric
                break
        if recommended_size is None:
            recommended_size = int(sizes[-1])
            primary_metric = 'max_size'

        stable_idx = next(
            (info['index'] for info in stability_points.values()
             if info['size'] == recommended_size and info.get('index') is not None),
            None
        )
        stable_pvalue_ks = stable_pvalue_chi2 = stable_rmse = None
        if stable_idx is not None:
            if 'ks_pvalue' in data_vars:
                stable_pvalue_ks = float(np.nanmedian(data_vars['ks_pvalue'][stable_idx, :]))
            if 'chi2_pvalue' in data_vars:
                stable_pvalue_chi2 = float(np.nanmedian(data_vars['chi2_pvalue'][stable_idx, :]))
            if 'rmse' in data_vars:
                stable_rmse = float(np.nanmedian(data_vars['rmse'][stable_idx, :]))

        # Optional figure
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
                    stability_method=stability_method,
                )
                if fig is not None:
                    fig.savefig(fig_output_path, dpi=150, bbox_inches='tight')
                    figure_path = fig_output_path
            except Exception as e:
                warnings.warn(f"Failed to create figure: {e}")

        import xarray as xr  # lazy import
        ds = xr.Dataset(
            data_vars={name: (['sizes', 'repeats'], arr) for name, arr in data_vars.items()},
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
                'created_by': 'MagicAdjuster.monte_carlo_fit',
            },
        )
        return ds

    # ------------------------------------------------------------------
    # Stability detection helpers (unchanged logic)
    # ------------------------------------------------------------------

    def _smooth_curve(self, x: np.ndarray, y: np.ndarray,
                      method: str = 'savgol') -> np.ndarray:
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < 4:
            return y
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        if method == 'savgol':
            window_length = min(len(x_valid), max(5, len(x_valid) // 3))
            if window_length % 2 == 0:
                window_length -= 1
            poly_order = min(3, window_length - 1)
            try:
                y_smooth = savgol_filter(y_valid, window_length, poly_order)
                result = np.full_like(y, np.nan)
                result[valid_mask] = y_smooth
                return result
            except Exception:
                return y
        elif method == 'spline':
            try:
                spl = UnivariateSpline(x_valid, y_valid, s=len(x_valid) * 0.1, k=3)
                result = np.full_like(y, np.nan)
                result[valid_mask] = spl(x_valid)
                return result
            except Exception:
                return y
        return y

    def _kneedle_detection(
        self, x: np.ndarray, y: np.ndarray,
        smooth: bool = True, smoothing_method: str = 'savgol',
    ) -> Tuple[Optional[int], Optional[np.ndarray]]:
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < 3:
            return None, None
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        y_smooth = None
        if smooth:
            y_work = self._smooth_curve(x, y, method=smoothing_method)
            y_work_valid = y_work[valid_mask]
            y_smooth = y_work
        else:
            y_work_valid = y_valid
        x_norm = (x_valid - x_valid.min()) / (x_valid.max() - x_valid.min() + 1e-10)
        y_norm = (y_work_valid - y_work_valid.min()) / (
            y_work_valid.max() - y_work_valid.min() + 1e-10)
        y_ref = np.linspace(y_norm[0], y_norm[-1], len(y_norm))
        if y_norm[-1] < y_norm[0]:
            differences = y_ref - y_norm
        else:
            differences = y_norm - y_ref
        if len(differences) == 0:
            return None, y_smooth
        knee_idx_local = np.argmax(differences)
        valid_indices = np.where(valid_mask)[0]
        return int(valid_indices[knee_idx_local]), y_smooth

    def _plateau_detection(
        self, x: np.ndarray, y: np.ndarray,
        consecutive_points: int = 3, relative_tolerance: float = 0.01,
    ) -> Optional[int]:
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < consecutive_points + 1:
            return None
        y_valid = y[valid_mask]
        rel_changes = []
        for i in range(1, len(y_valid)):
            if y_valid[i - 1] != 0:
                rel_changes.append(abs(y_valid[i] - y_valid[i - 1]) / abs(y_valid[i - 1]))
            else:
                rel_changes.append(np.inf)
        for i in range(len(rel_changes) - consecutive_points + 1):
            window = rel_changes[i:i + consecutive_points]
            if all(rc < relative_tolerance for rc in window if not np.isinf(rc)):
                valid_indices = np.where(valid_mask)[0]
                return int(valid_indices[i + 1])
        return None

    def _detect_stability_unified(
        self, data_vars: Dict[str, np.ndarray], sizes: np.ndarray,
        method: str = 'cv', **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        stability_points: Dict[str, Dict[str, Any]] = {}
        window_size = kwargs.get('window_size', max(2, len(sizes) // 4))
        cv_threshold = kwargs.get('cv_threshold', 0.1)
        smooth = kwargs.get('smooth', True)
        smoothing_method = kwargs.get('smoothing_method', 'savgol')
        consecutive_points = kwargs.get('consecutive_points', 3)
        relative_tolerance = kwargs.get('relative_tolerance', 0.01)

        for var_name, data_array in data_vars.items():
            if np.all(np.isnan(data_array)):
                stability_points[var_name] = {
                    'size': None, 'index': None,
                    'cv_at_stability': None, 'smoothed_curve': None, 'method': method,
                }
                continue

            medians = np.array([np.nanmedian(data_array[i, :]) for i in range(len(sizes))])
            stable_idx = None
            smoothed_curve = None
            cv_at_stability = None

            if method == 'cv':
                cv_values = []
                for i in range(len(sizes)):
                    size_data = data_array[i, ~np.isnan(data_array[i, :])]
                    if len(size_data) > 1:
                        mean_val = np.mean(size_data)
                        cv = np.std(size_data) / abs(mean_val) if mean_val != 0 else np.inf
                        cv_values.append(cv)
                    else:
                        cv_values.append(np.inf)
                for i in range(len(cv_values) - window_size + 1):
                    window_cvs = cv_values[i:i + window_size]
                    if all(cv < cv_threshold for cv in window_cvs if not np.isinf(cv)):
                        stable_idx = i
                        cv_at_stability = cv_values[i]
                        break
            elif method == 'kneedle':
                stable_idx, smoothed_curve = self._kneedle_detection(
                    sizes, medians, smooth=smooth, smoothing_method=smoothing_method)
            elif method == 'plateau':
                stable_idx = self._plateau_detection(
                    sizes, medians,
                    consecutive_points=consecutive_points,
                    relative_tolerance=relative_tolerance,
                )
            else:
                raise ValueError(f"Unknown stability method: {method!r}")

            if stable_idx is not None:
                stability_points[var_name] = {
                    'size': int(sizes[stable_idx]),
                    'index': int(stable_idx),
                    'cv_at_stability': cv_at_stability,
                    'smoothed_curve': smoothed_curve,
                    'method': method,
                }
            else:
                stability_points[var_name] = {
                    'size': None, 'index': None,
                    'cv_at_stability': None,
                    'smoothed_curve': smoothed_curve,
                    'method': method,
                }

        return stability_points

    def _detect_stability_points(self, data_vars, sizes):
        """Legacy alias; use _detect_stability_unified instead."""
        return self._detect_stability_unified(data_vars, sizes, method='cv')

    def _aggregate_and_detect_stability(
        self, results, sizes, test_list=None, max_params=None,
    ):
        """Aggregate MC results and detect stability (legacy aggregate method)."""
        summary: Dict[str, Any] = {'sizes': sizes.tolist(), 'tests': {}, 'params': {}}

        if test_list is None:
            inferred = set()
            for size in sizes:
                for rep in results.get('results', {}).get(int(size), []):
                    inferred.update(rep.get('gof', {}).keys())
                if inferred:
                    break
            test_list = sorted(inferred)

        if max_params is None:
            max_params = 0
            for size in sizes:
                for rep in results.get('results', {}).get(int(size), []):
                    p = rep.get('params')
                    if p is not None:
                        max_params = max(max_params, len(p))
                if max_params:
                    break

        tol, window = 0.1, 4
        for test in test_list:
            values_per_size = []
            for size in sizes:
                vals = []
                for rep in results['results'].get(int(size), []):
                    entry = rep.get('gof', {}).get(test)
                    if entry is None:
                        continue
                    if isinstance(entry, dict):
                        p = entry.get('p_value')
                        if p is not None:
                            vals.append(float(p))
                        else:
                            s = entry.get('chi2_statistic') or entry.get('ks_statistic')
                            if s is not None:
                                vals.append(float(s))
                    else:
                        try:
                            vals.append(float(entry))
                        except Exception:
                            pass
                values_per_size.append(vals)
            medians = np.array([np.median(v) if v else np.nan for v in values_per_size])
            inf_idx = None
            if medians.size >= window:
                for j in range(len(medians) - window + 1):
                    w = medians[j:j + window]
                    if np.all(np.isfinite(w)) and (w.max() - w.min() <= tol):
                        inf_idx = j + window - 1
                        break
            summary['tests'][test] = {
                'values_per_size': values_per_size,
                'medians': medians.tolist(),
                'inflection_index': int(inf_idx) if inf_idx is not None else None,
                'inflection_size': int(sizes[inf_idx]) if inf_idx is not None else None,
            }

        param_medians = {p: [] for p in range(max_params)}
        param_values_per_size = {p: [] for p in range(max_params)}
        for size in sizes:
            cols = {p: [] for p in range(max_params)}
            for rep in results['results'].get(int(size), []):
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
                param_medians[p].append(np.median(cols[p]) if cols[p] else np.nan)
        summary['params']['values_per_size'] = param_values_per_size
        summary['params']['medians'] = {p: list(v) for p, v in param_medians.items()}

        if max_params > 0:
            p0_meds = np.array(param_medians[0])
            rng_p0 = np.nanmax(p0_meds) - np.nanmin(p0_meds) if np.nanmax(p0_meds) != np.nanmin(p0_meds) else 1.0
            tol_p = 1e-3 * rng_p0
            inf_p = None
            if p0_meds.size >= window:
                for j in range(len(p0_meds) - window + 1):
                    w = p0_meds[j:j + window]
                    if np.all(np.isfinite(w)) and (w.max() - w.min() <= tol_p):
                        inf_p = j + window - 1
                        break
            summary['params']['inflection_index'] = int(inf_p) if inf_p is not None else None
            summary['params']['inflection_size'] = int(sizes[inf_p]) if inf_p is not None else None
        else:
            summary['params']['inflection_index'] = None
            summary['params']['inflection_size'] = None

        return summary, param_values_per_size, param_medians

    # ------------------------------------------------------------------
    # Figure generation (optional convenience)
    # ------------------------------------------------------------------

    def _create_monte_carlo_figure(
        self, data_vars, sizes, plot_type, stability_points=None,
        distribution_name=None, max_size=None, recommended_size=None,
        primary_metric=None, stable_pvalue_ks=None, stable_pvalue_chi2=None,
        stable_rmse=None, sampling_method=None, stability_method=None,
    ):
        """Create 2×3 summary figure (optional; returns None if matplotlib unavailable)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        param_names = sorted(k for k in data_vars if k.startswith('param_'))[:3]
        if not param_names:
            return None

        preferred_tests = ['ks_pvalue', 'chi2_pvalue', 'rmse']
        test_names = [t for t in preferred_tests if t in data_vars][:3]

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        axes_flat = axes.ravel()

        def _panel(ax, data: np.ndarray, title: str, var_name: str):
            if plot_type == 'boxplots':
                box_data = [data[i, ~np.isnan(data[i, :])] for i in range(len(sizes))]
                width = max(1, (sizes[1] - sizes[0]) * 0.6) if len(sizes) > 1 else 5
                bp = ax.boxplot(box_data, positions=sizes, widths=width, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
            else:
                med = np.nanmedian(data, axis=1)
                q25 = np.nanpercentile(data, 25, axis=1)
                q75 = np.nanpercentile(data, 75, axis=1)
                ax.plot(sizes, med, 'o-', lw=2, label='Median', color='steelblue', markersize=5)
                ax.fill_between(sizes, q25, q75, alpha=0.3, label='IQR (25-75%)', color='steelblue')
                if stability_points and var_name in stability_points:
                    sc = stability_points[var_name].get('smoothed_curve')
                    if sc is not None:
                        valid = ~np.isnan(sc)
                        if valid.any():
                            ax.plot(sizes[valid], sc[valid], '--', lw=1.5,
                                    label='Smoothed', color='orange', alpha=0.8)

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

            legend_items = []
            if recommended_size is not None:
                is_primary = (var_name == primary_metric)
                label = (f"Stable: n={recommended_size}" if is_primary
                         else f"RMSE stable: n={recommended_size}")
                vl = ax.axvline(recommended_size, color='red', linestyle='--',
                                linewidth=2, alpha=0.7, zorder=10)
                legend_items.append((vl, label))

            if stability_points and var_name in stability_points and var_name != primary_metric:
                sp = stability_points[var_name]
                if sp.get('size') is not None and sp['size'] != recommended_size:
                    vl = ax.axvline(sp['size'], color='gray', linestyle=':',
                                    linewidth=1.5, alpha=0.6, zorder=9)
                    legend_items.append((vl, f"{var_name} stable: n={sp['size']}"))

            if var_name in ('ks_pvalue', 'chi2_pvalue'):
                hl = ax.axhline(0.05, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7)
                legend_items.append((hl, 'α = 0.05'))
                pval = stable_pvalue_ks if var_name == 'ks_pvalue' else stable_pvalue_chi2
                if pval is not None:
                    ax.text(0.98, 0.98, f'p-value @ stable: {pval:.3f}',
                            transform=ax.transAxes, fontsize=9,
                            va='top', ha='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            if var_name == 'rmse' and stable_rmse is not None:
                ax.text(0.98, 0.98, f'RMSE @ stable: {stable_rmse:.4f}',
                        transform=ax.transAxes, fontsize=9, va='top', ha='right',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

            if plot_type == 'series' or legend_items:
                handles, labels = ax.get_legend_handles_labels()
                for h, l in legend_items:
                    handles.append(h)
                    labels.append(l)
                if handles:
                    ax.legend(handles, labels, frameon=True, fontsize=8,
                              loc='best', framealpha=0.9)

        for col, pname in enumerate(param_names):
            _panel(axes[0, col], data_vars[pname], pname.replace('_', ' ').title(), pname)
            axes[0, col].set_ylabel('Parameter Value', fontsize=10)
        for col in range(len(param_names), 3):
            axes[0, col].axis('off')

        for col, tname in enumerate(test_names):
            _panel(axes[1, col], data_vars[tname], tname.replace('_', ' ').title(), tname)
            axes[1, col].set_xlabel('Sample Size (n)', fontsize=10)
            if col == 0:
                axes[1, col].set_ylabel('Test Value', fontsize=10)
            if 'pvalue' in tname:
                try:
                    axes[1, col].set_ylim(-0.05, 1.05)
                except Exception:
                    pass
            elif tname == 'rmse':
                try:
                    axes[1, col].set_ylim(bottom=0)
                except Exception:
                    pass
        for col in range(len(test_names), 3):
            axes[1, col].axis('off')

        for ax in axes_flat:
            if ax.has_data():
                ax.tick_params(axis='both', labelsize=9)

        title_parts = ['Monte Carlo Stability Analysis']
        if distribution_name:
            title_parts.append(f'Distribution: {distribution_name}')
        if max_size:
            title_parts.append(f'Max n: {max_size}')
        if sampling_method:
            title_parts.append(f'Sampling: {sampling_method}')
        if stability_method:
            title_parts.append(f'Method: {stability_method.capitalize()}')
        if recommended_size and primary_metric:
            title_parts.append(f'Stable @ n={recommended_size} '
                               f'({primary_metric.replace("_", " ").upper()})')

        fig.suptitle(' | '.join(title_parts), fontsize=13, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def __repr__(self) -> str:
        dist = f", distribution='{self.distribution_name}'" if self.distribution_name else ''
        return f"MagicAdjuster(data_size={len(self.data)}{dist})"
