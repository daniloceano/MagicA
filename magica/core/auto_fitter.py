"""
Automatic distribution fitting with model selection
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from scipy import stats
import warnings

from .data_processor import DataProcessor
from .magic_adjuster import MagicAdjuster, FitResult, get_available_distributions


# ---------------------------------------------------------------------------
# Restricted families for extreme-value analysis
# ---------------------------------------------------------------------------

EVA_FAMILIES: Dict[str, List[str]] = {
    "bm":  ["genextreme", "gumbel_r"],   # Block Maxima
    "pot": ["genpareto", "expon"],        # Peaks over Threshold
}


class AutoFitter:
    """
    Automatic distribution fitting with model selection.

    Tests multiple candidate distributions and selects the best fit
    based on a user-supplied criterion.  Intended for the **standard
    statistics path only** (not for extreme value analysis, which should
    use :class:`~magica.core.extremes_analyzer.ExtremesAnalyzer` with the
    restricted families in :data:`EVA_FAMILIES`).

    All candidate :class:`~magica.core.magic_adjuster.FitResult` objects
    share the **same underlying data array** — no per-candidate copies are
    made.  Only scalar metrics (RMSE, AIC, BIC, p-values) and parameter
    tuples are retained per candidate; large objects are not accumulated.

    Parameters
    ----------
    data_processor : DataProcessor
    candidates : list of str, optional
        Distribution names to test.  Defaults to a curated subset of
        stable distributions suitable for environmental data.
    criterion : str, default ``'rmse'``
        Selection criterion: ``'rmse'``, ``'aic'``, ``'bic'``,
        ``'ks_pvalue'``, or ``'chi2_pvalue'``.

    Examples
    --------
    >>> import magica as ma, numpy as np
    >>> data = ma.read_data(np.random.weibull(2, 1000) * 8)
    >>> auto = data.get_auto_fitter()
    >>> best = auto.fit_best_distribution()
    >>> print(best.name, best.goodness_of_fit('rmse'))

    Batch/grid usage pattern
    ------------------------
    When applying AutoFitter to many spatial grid points, extract only the
    scalar results from each point and discard the heavy objects:

    >>> results = []
    >>> for point_data in grid_points:
    ...     proc = ma.read_data(point_data)
    ...     auto = proc.get_auto_fitter()
    ...     best = auto.fit_best_distribution()
    ...     results.append({
    ...         'name': best.name,
    ...         'params': best.params,
    ...         'rmse': best.goodness_of_fit('rmse'),
    ...     })
    ...     # best goes out of scope here; data array is freed
    """

    def __init__(
        self,
        data_processor: DataProcessor,
        candidates: Optional[List[str]] = None,
        criterion: str = 'rmse',
    ):
        if data_processor.data is None:
            raise ValueError("DataProcessor must contain data before auto-fitting.")

        self.data_processor = data_processor
        # Shared reference — no copy
        self.data: np.ndarray = data_processor.data

        all_distributions = get_available_distributions()

        if candidates is None:
            stable_defaults = [
                'weibull_min', 'lognorm', 'gamma', 'norm', 'expon', 'rayleigh',
                'chi2', 'beta', 'uniform', 'logistic', 'gumbel_r', 'pareto',
                'invgamma', 'maxwell', 'triang', 'laplace',
            ]
            self.candidates = [d for d in stable_defaults if d in all_distributions]
        else:
            invalid = [d for d in candidates if d not in all_distributions]
            if invalid:
                raise ValueError(f"Invalid distributions: {invalid}. "
                                 f"Available: {sorted(all_distributions.keys())}")
            self.candidates = list(candidates)

        self.criterion = criterion

        # Scalar-only cache: {name -> metrics dict}
        self._metrics: Dict[str, Dict[str, Any]] = {}
        # Best FitResult (only created once, at selection time)
        self._best_fit: Optional[FitResult] = None
        self._best_distribution: Optional[str] = None
        self._comparison_complete = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_single_distribution(self, distribution: str, **fit_kwargs) -> Dict[str, Any]:
        """
        Fit one distribution and return scalar metrics (no heavy objects stored).

        Parameters
        ----------
        distribution : str
        **fit_kwargs
            Passed to ``distribution.fit()``.

        Returns
        -------
        dict
            Keys: ``'distribution'``, ``'params'``, ``'rmse'``, ``'aic'``,
            ``'bic'``, ``'ks_statistic'``, ``'ks_pvalue'``,
            ``'chi2_statistic'``, ``'chi2_pvalue'``, ``'success'``.
        """
        try:
            adjuster = MagicAdjuster(self.data_processor)
            fit = adjuster.fit_distribution(distribution, **fit_kwargs)

            rmse = fit.goodness_of_fit('rmse')
            aic = fit.goodness_of_fit('aic')
            bic = fit.goodness_of_fit('bic')
            ks = fit.goodness_of_fit('ks')
            chi2 = fit.goodness_of_fit('chi2', warn_on_normalization=False)

            result = {
                'distribution': distribution,
                'params': fit.params,
                'rmse': rmse,
                'aic': aic,
                'bic': bic,
                'ks_statistic': ks.get('ks_statistic', np.nan),
                'ks_pvalue': ks.get('p_value', np.nan),
                'chi2_statistic': chi2.get('chi2_statistic', np.nan),
                'chi2_pvalue': chi2.get('p_value', np.nan),
                'success': True,
            }
            # Scalar-only cache; the FitResult itself is not retained
            self._metrics[distribution] = result
            return result

        except Exception as e:
            error_result: Dict[str, Any] = {
                'distribution': distribution,
                'params': None,
                'rmse': float('inf'),
                'aic': float('inf'),
                'bic': float('inf'),
                'ks_statistic': np.nan,
                'ks_pvalue': np.nan,
                'chi2_statistic': np.nan,
                'chi2_pvalue': np.nan,
                'success': False,
                'error': str(e),
            }
            self._metrics[distribution] = error_result
            warnings.warn(f"Failed to fit {distribution}: {e}")
            return error_result

    def fit_all_distributions(self, **fit_kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Fit all candidates and return scalar metrics.

        Returns
        -------
        dict
            ``{distribution_name: metrics_dict}``
        """
        print(f"Testing {len(self.candidates)} distributions...")
        for i, dist in enumerate(self.candidates, 1):
            print(f"  [{i}/{len(self.candidates)}] Fitting {dist}...")
            self.fit_single_distribution(dist, **fit_kwargs)
        self._comparison_complete = True
        print("✓ All distributions fitted successfully")
        return dict(self._metrics)

    def fit_best_distribution(self, **fit_kwargs) -> FitResult:
        """
        Select the best distribution and return its :class:`FitResult`.

        Fits all candidates if not already done.

        Returns
        -------
        FitResult
            The fit for the best-performing distribution.
        """
        if not self._comparison_complete:
            self.fit_all_distributions(**fit_kwargs)

        valid = {k: v for k, v in self._metrics.items() if v['success']}
        if not valid:
            raise RuntimeError("No distributions fitted successfully.")

        if self.criterion == 'rmse':
            best_name = min(valid, key=lambda x: valid[x]['rmse'])
        elif self.criterion == 'aic':
            best_name = min(valid, key=lambda x: valid[x]['aic'])
        elif self.criterion == 'bic':
            best_name = min(valid, key=lambda x: valid[x]['bic'])
        elif self.criterion == 'ks_pvalue':
            best_name = max(valid, key=lambda x: valid[x]['ks_pvalue'])
        elif self.criterion == 'chi2_pvalue':
            best_name = max(valid, key=lambda x: valid[x]['chi2_pvalue'])
        else:
            raise ValueError(f"Unknown criterion: {self.criterion!r}")

        self._best_distribution = best_name

        # Create the FitResult once at selection time
        adjuster = MagicAdjuster(self.data_processor)
        self._best_fit = adjuster.fit_distribution(best_name, **fit_kwargs)
        return self._best_fit

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_comparison_table(self, sort_by: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Return sorted comparison table of scalar metrics.

        Parameters
        ----------
        sort_by : str, optional
            Metric to sort by (defaults to ``criterion``).

        Returns
        -------
        dict
        """
        if not self._comparison_complete:
            raise RuntimeError("Call fit_all_distributions() first.")
        sort_key = sort_by or self.criterion
        reverse = sort_key in ('ks_pvalue', 'chi2_pvalue')
        return dict(
            sorted(self._metrics.items(),
                   key=lambda x: x[1].get(sort_key, float('inf')),
                   reverse=reverse)
        )

    def get_best_adjuster(self) -> MagicAdjuster:
        """
        Return a :class:`MagicAdjuster` fitted with the best distribution.

        .. deprecated::
            Prefer ``fit_best_distribution()`` which returns a
            :class:`FitResult` directly.
        """
        warnings.warn(
            "get_best_adjuster() is deprecated; use fit_best_distribution() "
            "which returns a FitResult directly.",
            DeprecationWarning, stacklevel=2,
        )
        if self._best_distribution is None:
            self.fit_best_distribution()
        adjuster = MagicAdjuster(self.data_processor)
        adjuster.fit_distribution(self._best_distribution)
        return adjuster

    @staticmethod
    def get_all_available_distributions() -> List[str]:
        """Return sorted list of all available distribution names."""
        return sorted(get_available_distributions().keys())

    def __repr__(self) -> str:
        status = "fitted" if self._comparison_complete else "not fitted"
        best = f", best={self._best_distribution}" if self._best_distribution else ""
        return (f"AutoFitter(candidates={len(self.candidates)}, "
                f"criterion={self.criterion}, {status}{best})")
