"""
Extreme value analysis: Block Maxima and Peaks over Threshold

Provides return-period and return-value calculations with correct formulas
for both Block Maxima (GEV) and Peaks-over-Threshold (GPD) methods.

GPD convention used throughout:
  scipy.stats.genpareto is fitted to exceedances **above the threshold**
  (i.e. ``exceedances - u``) with ``floc=0``, yielding params ``(ξ, 0, σ)``.

Return-level formulas:
  Block Maxima:
    x_T = ppf(1 - 1/T)   where T is in block units (typically years)
  PoT + GPD:
    if |ξ| >= 1e-6:  x_T = u + (σ/ξ) * ((λ*T)^ξ - 1)
    if |ξ| <  1e-6:  x_T = u + σ * ln(λ*T)
  where λ = n_independent / time_span  (exceedances per year)
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, Tuple, List
from scipy import stats

from .data_processor import DataProcessor
from .magic_adjuster import FitResult, get_available_distributions
from .auto_fitter import EVA_FAMILIES


# ---------------------------------------------------------------------------
# EVAFit – FitResult extended with extremes metadata
# ---------------------------------------------------------------------------

@dataclass(eq=False, frozen=True)
class EVAFit(FitResult):
    """
    Immutable result of an extreme-value fit, augmented with EVA metadata.

    All :class:`~magica.core.magic_adjuster.FitResult` methods are available.

    Additional attributes
    ---------------------
    method : str
        ``'bm'`` (Block Maxima) or ``'pot'`` (Peaks over Threshold).
    threshold : float or None
        PoT threshold ``u``.  ``None`` for Block Maxima.
    lambda_rate : float or None
        Exceedance rate λ (independent exceedances per year).  ``None`` for BM.
    blocks_per_year : float or None
        For Block Maxima: number of blocks per year (typically 1).  ``None`` for PoT.
    """

    method: str
    threshold: Optional[float]
    lambda_rate: Optional[float]
    blocks_per_year: Optional[float]

    def return_value(self, T: Union[float, np.ndarray]) -> np.ndarray:
        """
        Return level for return period *T* (in years).

        Dispatches to the correct formula based on ``method``.

        Parameters
        ----------
        T : float or array-like

        Returns
        -------
        ndarray
        """
        T = np.asarray(T, dtype=float)
        if self.method == 'bm':
            return self._rv_block_maxima(T)
        elif self.method == 'pot':
            return self._rv_pot(T)
        else:
            raise ValueError(f"Unknown EVA method: {self.method!r}")

    def return_period(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Return period for value *x* (in years).

        Parameters
        ----------
        x : float or array-like

        Returns
        -------
        ndarray
        """
        x = np.asarray(x, dtype=float)
        if self.method == 'bm':
            return self._rp_block_maxima(x)
        elif self.method == 'pot':
            return self._rp_pot(x)
        else:
            raise ValueError(f"Unknown EVA method: {self.method!r}")

    def return_level_table(
        self, periods: List[float]
    ) -> List[Dict[str, float]]:
        """
        Serialisable table of return periods and corresponding levels.

        Suitable for passing to a frontend.  Confidence intervals are
        reserved as ``None`` (placeholder for future bootstrap CI support).

        Parameters
        ----------
        periods : list of float

        Returns
        -------
        list of dict
            ``[{'period': T, 'level': x, 'ci_low': None, 'ci_high': None}, ...]``
        """
        levels = self.return_value(np.asarray(periods, dtype=float))
        return [
            {'period': float(T), 'level': float(rv), 'ci_low': None, 'ci_high': None}
            for T, rv in zip(periods, np.atleast_1d(levels))
        ]

    # ------------------------------------------------------------------
    # Private: BM formulas
    # ------------------------------------------------------------------

    def _rv_block_maxima(self, T: np.ndarray) -> np.ndarray:
        """x_T = ppf(1 - 1/T)"""
        non_exc = 1.0 - 1.0 / T
        return self.distribution.ppf(non_exc, *self.params)

    def _rp_block_maxima(self, x: np.ndarray) -> np.ndarray:
        """T = 1 / (1 - CDF(x))"""
        exc_prob = 1.0 - self.distribution.cdf(x, *self.params)
        return np.where(exc_prob > 0, 1.0 / exc_prob, np.inf)

    # ------------------------------------------------------------------
    # Private: PoT + GPD formulas
    # ------------------------------------------------------------------

    def _rv_pot(self, T: np.ndarray) -> np.ndarray:
        """
        Return level for PoT + GPD.

        Convention: genpareto fitted to (exceedances - u) with floc=0 so that
        params = (ξ, 0, σ).  Params tuple from scipy: (c, loc, scale) = (ξ, 0, σ).
        """
        u = self.threshold
        lam = self.lambda_rate
        if u is None or lam is None:
            raise ValueError("threshold and lambda_rate must be set for PoT return values.")

        xi = float(self.params[0])   # shape (ξ)
        sigma = float(self.params[2])  # scale (σ); params[1] = loc = 0 (floc=0)

        lam_T = lam * T

        if abs(xi) >= 1e-6:
            return u + (sigma / xi) * (lam_T ** xi - 1.0)
        else:
            # Gumbel / exponential limit (ξ → 0)
            return u + sigma * np.log(lam_T)

    def _rp_pot(self, x: np.ndarray) -> np.ndarray:
        """
        Return period for PoT + GPD.

        Computes the exceedance rate of *x* via the GPD tail, then T = 1/λ_x.
        """
        u = self.threshold
        lam = self.lambda_rate
        if u is None or lam is None:
            raise ValueError("threshold and lambda_rate must be set for PoT return periods.")

        xi = float(self.params[0])
        sigma = float(self.params[2])
        z = x - u

        if abs(xi) >= 1e-6:
            lam_x = lam * (1.0 + xi * z / sigma) ** (-1.0 / xi)
        else:
            lam_x = lam * np.exp(-z / sigma)

        return np.where(lam_x > 0, 1.0 / lam_x, np.inf)

    def __repr__(self) -> str:
        param_str = ", ".join(f"{p:.4g}" for p in self.params)
        meta = (f"method='{self.method}', "
                f"threshold={self.threshold}, lambda={self.lambda_rate:.4g}"
                if self.method == 'pot' and self.lambda_rate is not None
                else f"method='{self.method}'")
        return (f"EVAFit(distribution='{self.name}', "
                f"params=({param_str}), {meta}, data_size={len(self.data)})")


# ---------------------------------------------------------------------------
# ExtremesAnalyzer
# ---------------------------------------------------------------------------

class ExtremesAnalyzer:
    """
    Extreme value analysis with return period and return value calculations.

    Supports Block Maxima (BM) and Peaks over Threshold (PoT) methods.
    Both methods fit distributions **to the extracted extreme sample**, not
    to the full time series.

    Parameters
    ----------
    data_processor : DataProcessor
    times : array-like, optional
    time_unit : str, default ``'years'``

    Examples
    --------
    **Block Maxima (GEV):**

    >>> extremes = processor.get_extremes_analyzer()
    >>> annual_max, times = extremes.extract_block_maxima('YE')
    >>> ev_fit = extremes.fit_block_maxima(annual_max, times)
    >>> ev_fit.return_value([10, 50, 100])

    **Peaks over Threshold (GPD):**

    >>> result = extremes.find_optimal_pot_threshold(min_samples=50)
    >>> ev_fit = extremes.fit_pot(result)
    >>> ev_fit.return_value([10, 50, 100])
    """

    def __init__(
        self,
        data_processor: DataProcessor,
        times: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]] = None,
        time_unit: str = 'years',
    ):
        if data_processor.data is None:
            raise ValueError("DataProcessor must contain data before extreme analysis.")

        self.data_processor = data_processor
        self.data = data_processor.data  # shared reference
        self.time_unit = time_unit

        self._process_times(times)

        # Legacy state — populated by fit_distribution() for backward compat
        self.distribution_name: Optional[str] = None
        self.fitted_params: Optional[tuple] = None
        self._eva_fit: Optional[EVAFit] = None

    # ------------------------------------------------------------------
    # Time handling
    # ------------------------------------------------------------------

    def _process_times(
        self, times: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]]
    ):
        if times is None:
            if hasattr(self.data_processor, '_original_data'):
                orig = self.data_processor._original_data
                if isinstance(orig, pd.Series) and isinstance(orig.index, pd.DatetimeIndex):
                    times = orig.index

        if times is None:
            warnings.warn(
                "No time information provided. Assuming uniform time spacing. "
                "Return periods will be in units of observation count."
            )
            self.times = np.arange(len(self.data))
            self.has_datetime = False
            self.time_span = float(len(self.data))
        elif isinstance(times, pd.DatetimeIndex):
            self.times = times
            self.has_datetime = True
            self.time_span = self._calculate_time_span(times)
        elif isinstance(times, pd.Series):
            if pd.api.types.is_datetime64_any_dtype(times):
                self.times = pd.DatetimeIndex(times)
                self.has_datetime = True
                self.time_span = self._calculate_time_span(self.times)
            else:
                self.times = times.values
                self.has_datetime = False
                self.time_span = float(np.ptp(self.times))
        else:
            times_array = np.array(times)
            if np.issubdtype(times_array.dtype, np.datetime64):
                self.times = pd.DatetimeIndex(times_array)
                self.has_datetime = True
                self.time_span = self._calculate_time_span(self.times)
            else:
                self.times = times_array
                self.has_datetime = False
                self.time_span = float(np.ptp(times_array)) if len(times_array) > 1 else float(len(times_array))

    def _calculate_time_span(self, times: pd.DatetimeIndex) -> float:
        delta = times[-1] - times[0]
        if self.time_unit == 'years':
            return delta.total_seconds() / (365.25 * 24 * 3600)
        elif self.time_unit == 'days':
            return delta.total_seconds() / (24 * 3600)
        elif self.time_unit == 'hours':
            return delta.total_seconds() / 3600
        elif self.time_unit == 'months':
            return delta.total_seconds() / (30.44 * 24 * 3600)
        else:
            raise ValueError(f"Unknown time unit: {self.time_unit!r}")

    # ------------------------------------------------------------------
    # Primary EVA fitting methods
    # ------------------------------------------------------------------

    def fit_block_maxima(
        self,
        block_maxima: np.ndarray,
        times: Optional[pd.DatetimeIndex] = None,
        distribution: str = 'genextreme',
        blocks_per_year: float = 1.0,
        **fit_kwargs,
    ) -> EVAFit:
        """
        Fit a distribution to block maxima and return an :class:`EVAFit`.

        The fit is performed on *block_maxima*, not on the full series.

        Parameters
        ----------
        block_maxima : ndarray
            Values extracted by :meth:`extract_block_maxima`.
        times : DatetimeIndex, optional
            Block times (not used in fitting, kept for record).
        distribution : str, default ``'genextreme'``
            Must be in ``EVA_FAMILIES['bm']``: ``'genextreme'`` or ``'gumbel_r'``.
        blocks_per_year : float, default 1.0
            Number of blocks per year (used to document the fit; does not
            change the return-level formula for BM).
        **fit_kwargs
            Passed to ``distribution.fit()``.

        Returns
        -------
        EVAFit
        """
        _check_eva_family('bm', distribution)

        dist_map = get_available_distributions()
        dist_obj = dist_map[distribution.lower()]
        params = dist_obj.fit(block_maxima, **fit_kwargs)

        eva_fit = EVAFit(
            distribution=dist_obj,
            name=distribution.lower(),
            params=tuple(params),
            data=block_maxima,
            method='bm',
            threshold=None,
            lambda_rate=None,
            blocks_per_year=blocks_per_year,
        )
        self._eva_fit = eva_fit
        self.distribution_name = eva_fit.name
        self.fitted_params = eva_fit.params
        return eva_fit

    def fit_pot(
        self,
        pot_result: Dict[str, Any],
        distribution: str = 'genpareto',
    ) -> EVAFit:
        """
        Fit a GPD to PoT exceedances and return an :class:`EVAFit`.

        The fit uses exceedances **above the threshold** (``exceedances - u``)
        with ``floc=0``, yielding parameters ``(ξ, 0, σ)``.

        Parameters
        ----------
        pot_result : dict
            Dictionary as returned by :meth:`find_optimal_pot_threshold`.
            Must contain ``'exceedances'``, ``'threshold'``, and
            ``'n_independent'``.
        distribution : str, default ``'genpareto'``
            Must be in ``EVA_FAMILIES['pot']``.

        Returns
        -------
        EVAFit
        """
        _check_eva_family('pot', distribution)

        exceedances = pot_result['exceedances']
        u = float(pot_result['threshold'])
        n_independent = int(pot_result['n_independent'])

        if len(exceedances) == 0:
            raise ValueError("No exceedances in pot_result; cannot fit distribution.")
        if self.time_span <= 0:
            raise ValueError("time_span must be > 0 to compute exceedance rate λ.")

        # λ = independent exceedances per year
        lambda_rate = n_independent / self.time_span

        dist_map = get_available_distributions()
        dist_obj = dist_map[distribution.lower()]

        # Fit to exceedances - u with floc=0
        exceedances_above = exceedances - u
        params = dist_obj.fit(exceedances_above, floc=0)

        eva_fit = EVAFit(
            distribution=dist_obj,
            name=distribution.lower(),
            params=tuple(params),
            data=exceedances_above,
            method='pot',
            threshold=u,
            lambda_rate=lambda_rate,
            blocks_per_year=None,
        )
        self._eva_fit = eva_fit
        self.distribution_name = eva_fit.name
        self.fitted_params = eva_fit.params
        return eva_fit

    # ------------------------------------------------------------------
    # Legacy fit_distribution (backward compat for BM path only)
    # ------------------------------------------------------------------

    def fit_distribution(
        self, distribution: Union[str, object], **kwargs
    ) -> 'ExtremesAnalyzer':
        """
        Fit a distribution directly on the analyzer's current data.

        .. deprecated::
            Prefer :meth:`fit_block_maxima` or :meth:`fit_pot` which fit on
            the *extracted* extreme sample and carry the correct metadata for
            return-level calculations.

        This legacy method fits on the data held by the internal
        DataProcessor (which may be the full series if the caller already
        created a processor from the extracted extremes), using the BM
        return-level formula.  It is retained for backward compatibility
        with existing tutorials and notebooks.

        Returns
        -------
        ExtremesAnalyzer
            Self for method chaining.
        """
        if isinstance(distribution, str):
            dist_map = get_available_distributions()
            name = distribution.lower()
            if name not in dist_map:
                raise ValueError(f"Unknown distribution {distribution!r}.")
            dist_obj = dist_map[name]
        else:
            dist_obj = distribution
            name = getattr(distribution, 'name', str(distribution))

        params = dist_obj.fit(self.data, **kwargs)

        # Wrap as BM EVAFit for consistency
        self._eva_fit = EVAFit(
            distribution=dist_obj,
            name=name,
            params=tuple(params),
            data=self.data,
            method='bm',
            threshold=None,
            lambda_rate=None,
            blocks_per_year=1.0,
        )
        self.distribution_name = name
        self.fitted_params = tuple(params)
        return self

    # ------------------------------------------------------------------
    # Return-level calculations (delegate to EVAFit)
    # ------------------------------------------------------------------

    def return_value(
        self, return_period: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Return value for the given return period(s).

        Dispatches to the correct formula (BM or PoT) based on the last
        fitted :class:`EVAFit`.

        Parameters
        ----------
        return_period : float or array-like
            In ``time_unit`` units (typically years).

        Returns
        -------
        ndarray
        """
        if self._eva_fit is None:
            raise ValueError(
                "Must fit a distribution first. "
                "Use fit_block_maxima(), fit_pot(), or fit_distribution()."
            )
        return self._eva_fit.return_value(return_period)

    def return_period(self, value: Union[float, np.ndarray]) -> np.ndarray:
        """
        Return period for the given value(s).

        Parameters
        ----------
        value : float or array-like

        Returns
        -------
        ndarray
        """
        if self._eva_fit is None:
            raise ValueError(
                "Must fit a distribution first. "
                "Use fit_block_maxima(), fit_pot(), or fit_distribution()."
            )
        return self._eva_fit.return_period(value)

    def return_level_table(self, periods: List[float]) -> List[Dict[str, float]]:
        """
        Serialisable table of return periods and levels (for frontends).

        Parameters
        ----------
        periods : list of float

        Returns
        -------
        list of dict
        """
        if self._eva_fit is None:
            raise ValueError("Must fit a distribution first.")
        return self._eva_fit.return_level_table(periods)

    # ------------------------------------------------------------------
    # Distribution evaluation helpers
    # ------------------------------------------------------------------

    def ppf(self, q):
        if self._eva_fit is None:
            raise ValueError("Must fit a distribution first.")
        return self._eva_fit.ppf(q)

    def cdf(self, x):
        if self._eva_fit is None:
            raise ValueError("Must fit a distribution first.")
        return self._eva_fit.cdf(x)

    def pdf(self, x):
        if self._eva_fit is None:
            raise ValueError("Must fit a distribution first.")
        return self._eva_fit.pdf(x)

    def goodness_of_fit(self, method: str, **kwargs):
        if self._eva_fit is None:
            raise ValueError("Must fit a distribution first.")
        return self._eva_fit.goodness_of_fit(method, **kwargs)

    # ------------------------------------------------------------------
    # Extraction methods
    # ------------------------------------------------------------------

    def extract_block_maxima(
        self,
        block_size: str = 'YE',
        method: str = 'max',
    ) -> Tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
        """
        Extract block maxima (or minima) from the time series.

        Parameters
        ----------
        block_size : str, default ``'YE'``
            Pandas offset alias: ``'YE'``/``'Y'`` (annual), ``'QE'`` (quarterly),
            ``'ME'`` (monthly), ``'W'`` (weekly), ``'D'`` (daily).
        method : str, default ``'max'``
            ``'max'`` or ``'min'``.

        Returns
        -------
        values : ndarray
        times : DatetimeIndex or None
        """
        if not self.has_datetime:
            raise ValueError(
                "Block maxima extraction requires datetime information."
            )
        series = pd.Series(self.data, index=self.times)
        if method == 'max':
            resampled = series.resample(block_size).max()
        elif method == 'min':
            resampled = series.resample(block_size).min()
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'max' or 'min'.")
        resampled = resampled.dropna()
        return resampled.values, resampled.index

    def peaks_over_threshold(
        self,
        threshold: float,
        min_separation: Optional[Union[str, pd.Timedelta, int, float]] = None,
        event_wise: bool = False,
    ) -> Tuple[np.ndarray, Optional[Union[pd.DatetimeIndex, np.ndarray]]]:
        """
        Extract peaks over threshold (PoT) from the time series.

        Parameters
        ----------
        threshold : float
        min_separation : str, Timedelta, int, or float, optional
            Minimum time between peaks.  Numeric values are interpreted as days.
        event_wise : bool, default False
            If True, return one peak per consecutive exceedance event.

        Returns
        -------
        exceedances : ndarray
        times : DatetimeIndex or ndarray or None
        """
        exceed_mask = self.data > threshold

        if not self.has_datetime:
            if min_separation is not None or event_wise:
                warnings.warn(
                    "min_separation and event_wise require datetime information. "
                    "Returning all exceedances without declustering."
                )
            return self.data[exceed_mask], None

        if event_wise:
            series = pd.Series(self.data, index=self.times)
            exceed_series = series[exceed_mask]
            if len(exceed_series) == 0:
                return np.array([]), pd.DatetimeIndex([])
            time_diffs = exceed_series.index.to_series().diff()
            original_diffs = pd.Series(self.times).diff().dropna()
            expected_step = original_diffs.mode()[0] if len(original_diffs) > 0 else pd.Timedelta(days=1)
            is_new_event = time_diffs > expected_step
            event_ids = is_new_event.cumsum()
            maxima, times_out = [], []
            for eid in event_ids.unique():
                event_data = exceed_series[event_ids == eid]
                idx = event_data.idxmax()
                maxima.append(event_data[idx])
                times_out.append(idx)
            return np.array(maxima), pd.DatetimeIndex(times_out)

        exceedances = self.data[exceed_mask]
        exceed_times = self.times[exceed_mask]

        if min_separation is None:
            return exceedances, exceed_times

        if isinstance(min_separation, str):
            min_separation = pd.Timedelta(min_separation)
        elif isinstance(min_separation, (int, float)):
            min_separation = pd.Timedelta(days=min_separation)

        keep = [0]
        for i in range(1, len(exceed_times)):
            if exceed_times[i] - exceed_times[keep[-1]] >= min_separation:
                keep.append(i)

        return exceedances[keep], exceed_times[keep]

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Basic summary statistics of the full dataset."""
        summary = {
            'n_observations': len(self.data),
            'time_span': self.time_span,
            'time_unit': self.time_unit,
            'max': float(np.max(self.data)),
            'min': float(np.min(self.data)),
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data)),
            'percentile_95': float(np.percentile(self.data, 95)),
            'percentile_99': float(np.percentile(self.data, 99)),
            'has_datetime': self.has_datetime,
            'distribution_name': self.distribution_name,
        }
        if self.has_datetime:
            summary['start_date'] = str(self.times[0])
            summary['end_date'] = str(self.times[-1])
        if self.fitted_params is not None:
            summary['distribution_params'] = self.fitted_params
        return summary

    # ------------------------------------------------------------------
    # Threshold search
    # ------------------------------------------------------------------

    def find_optimal_pot_threshold(
        self,
        min_samples: int = 50,
        percentile_min: float = 90,
        percentile_max: float = 99,
        percentile_step: float = 1.0,
        min_separation_hours: float = 48,
        max_separation_hours: float = 120,
        separation_step_hours: float = 24,
        vary_first: str = 'percentile',
        max_iterations: int = 200,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for a PoT threshold yielding at least *min_samples* independent peaks.

        Parameters
        ----------
        min_samples : int, default 50
        percentile_min : float, default 90
        percentile_max : float, default 99
        percentile_step : float, default 1.0
        min_separation_hours : float, default 48
        max_separation_hours : float, default 120
        separation_step_hours : float, default 24
        vary_first : str, default ``'percentile'``
            ``'percentile'`` or ``'separation'``.
        max_iterations : int, default 200
        verbose : bool, default False

        Returns
        -------
        dict
            Keys: ``'threshold'``, ``'percentile'``, ``'separation_hours'``,
            ``'n_raw_exceedances'``, ``'n_independent'``, ``'exceedances'``,
            ``'exceedance_times'``, ``'success'``, ``'iterations'``.
        """
        if not self.has_datetime:
            raise ValueError(
                "POT threshold search requires datetime information."
            )
        if vary_first not in ('percentile', 'separation'):
            raise ValueError("vary_first must be 'percentile' or 'separation'.")
        if percentile_min >= percentile_max:
            raise ValueError("percentile_min must be < percentile_max.")
        if min_separation_hours >= max_separation_hours:
            raise ValueError("min_separation_hours must be < max_separation_hours.")

        iteration = 0
        best_result = None

        if vary_first == 'percentile':
            separations = np.arange(min_separation_hours, max_separation_hours + 1,
                                    separation_step_hours)
            for sep_h in separations:
                if verbose:
                    print(f"\n--- separation: {sep_h}h ---")
                current_pct = percentile_max
                while current_pct >= percentile_min:
                    iteration += 1
                    if iteration > max_iterations:
                        break
                    u = float(np.percentile(self.data, current_pct))
                    n_raw = int((self.data > u).sum())
                    if n_raw == 0:
                        current_pct -= percentile_step
                        continue
                    try:
                        exc, exc_t = self.peaks_over_threshold(
                            threshold=u, min_separation=sep_h / 24)
                        n_ind = len(exc)
                        result = {
                            'threshold': u,
                            'percentile': current_pct,
                            'separation_hours': sep_h,
                            'n_raw_exceedances': n_raw,
                            'n_independent': n_ind,
                            'exceedances': exc,
                            'exceedance_times': exc_t,
                            'success': n_ind >= min_samples,
                            'iterations': iteration,
                        }
                        if best_result is None or n_ind > best_result['n_independent']:
                            best_result = result.copy()
                        if verbose:
                            print(f"{'✓' if result['success'] else '•'} "
                                  f"p={current_pct:.1f}, thresh={u:.2f}, n={n_ind}")
                        if n_ind >= min_samples:
                            return result
                    except Exception as e:
                        if verbose:
                            print(f"✗ p={current_pct:.1f}: {e}")
                    current_pct -= percentile_step
                if iteration > max_iterations:
                    break

        else:  # vary separation first
            percentiles = np.arange(percentile_max, percentile_min - percentile_step,
                                    -percentile_step)
            for pct in percentiles:
                if verbose:
                    print(f"\n--- percentile: {pct:.1f} ---")
                current_sep = min_separation_hours
                while current_sep <= max_separation_hours:
                    iteration += 1
                    if iteration > max_iterations:
                        break
                    u = float(np.percentile(self.data, pct))
                    n_raw = int((self.data > u).sum())
                    if n_raw == 0:
                        break
                    try:
                        exc, exc_t = self.peaks_over_threshold(
                            threshold=u, min_separation=current_sep / 24)
                        n_ind = len(exc)
                        result = {
                            'threshold': u,
                            'percentile': pct,
                            'separation_hours': current_sep,
                            'n_raw_exceedances': n_raw,
                            'n_independent': n_ind,
                            'exceedances': exc,
                            'exceedance_times': exc_t,
                            'success': n_ind >= min_samples,
                            'iterations': iteration,
                        }
                        if best_result is None or n_ind > best_result['n_independent']:
                            best_result = result.copy()
                        if verbose:
                            print(f"{'✓' if result['success'] else '•'} "
                                  f"sep={current_sep}h, thresh={u:.2f}, n={n_ind}")
                        if n_ind >= min_samples:
                            return result
                    except Exception as e:
                        if verbose:
                            print(f"✗ sep={current_sep}h: {e}")
                    current_sep += separation_step_hours
                if iteration > max_iterations:
                    break

        # No solution meeting min_samples
        if best_result is None:
            return {
                'threshold': np.nan, 'percentile': np.nan, 'separation_hours': np.nan,
                'n_raw_exceedances': 0, 'n_independent': 0,
                'exceedances': np.array([]),
                'exceedance_times': pd.DatetimeIndex([]) if self.has_datetime else None,
                'success': False, 'iterations': iteration,
                'warning': 'No exceedances found in search range',
            }

        best_result['warning'] = (
            f"Could not find threshold with {min_samples} samples. "
            f"Best: {best_result['n_independent']} samples. Relax constraints."
        )
        if verbose:
            print(f"\n⚠️  {best_result['warning']}")
        return best_result

    # ------------------------------------------------------------------
    # Visualisation (optional convenience)
    # ------------------------------------------------------------------

    def plot_return_levels(
        self,
        ax=None,
        return_periods: Optional[np.ndarray] = None,
        empirical: bool = True,
        title: Optional[str] = None,
    ):
        """
        Plot return level diagram (convenience wrapper).

        Parameters
        ----------
        ax : matplotlib Axes, optional
        return_periods : array-like, optional
        empirical : bool, default True
        title : str, optional

        Returns
        -------
        (fig, ax)
        """
        import matplotlib.pyplot as plt  # lazy import

        if self._eva_fit is None:
            raise ValueError("Must fit a distribution before plotting.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = None

        if return_periods is None:
            return_periods = np.logspace(0, 3, 50)

        theoretical_rv = self.return_value(return_periods)
        ax.plot(return_periods, theoretical_rv, 'r-', linewidth=2,
                label=f'Theoretical ({self.distribution_name})')

        if empirical:
            sorted_data = np.sort(self.data)[::-1]
            n = len(sorted_data)
            empirical_rp = (n + 1) / np.arange(1, n + 1)
            ax.plot(empirical_rp, sorted_data, 'bo', alpha=0.6,
                    markersize=4, label='Empirical')

        ax.set_xlabel(f'Return Period ({self.time_unit})')
        ax.set_ylabel('Return Value')
        ax.set_title(title or 'Return Level Plot')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()

        if fig is not None:
            plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_directional_return_values(
        directional_results: Dict[str, Dict[str, Any]],
        return_periods: list = None,
        colors: list = None,
        overlay: bool = False,
        show_values: bool = True,
        figsize: tuple = None,
        title: str = None,
    ) -> tuple:
        """
        Polar plots of return values by direction.

        Parameters
        ----------
        directional_results : dict
        return_periods : list, optional
        colors : list, optional
        overlay : bool, default False
        show_values : bool, default True
        figsize : tuple, optional
        title : str, optional

        Returns
        -------
        (fig, axes)
        """
        import matplotlib.pyplot as plt  # lazy import

        if not directional_results:
            raise ValueError("directional_results cannot be empty.")
        if return_periods is None:
            return_periods = [10, 20, 50, 100]
        if len(return_periods) > 4:
            raise ValueError("Maximum 4 return periods can be plotted.")
        if not return_periods:
            raise ValueError("At least one return period must be specified.")
        if colors is None:
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(return_periods)]
        elif len(colors) != len(return_periods):
            raise ValueError("Number of colors must match number of return periods.")

        sector_data = [
            (name, res['center_deg'])
            for name, res in directional_results.items()
            if 'center_deg' in res
        ]
        sector_data.sort(key=lambda x: x[1])
        sector_names = [n for n, _ in sector_data]
        if not sector_names:
            raise ValueError("No sectors found in directional_results.")

        n_periods = len(return_periods)

        if overlay:
            figsize = figsize or (10, 10)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='polar')
            axes = ax
            for i, rp in enumerate(return_periods):
                angles, values = [], []
                for sn in sector_names:
                    res = directional_results[sn]
                    if res.get('success') and 'return_values' in res:
                        rv = res['return_values'].get(rp)
                        if rv is not None and not np.isnan(rv):
                            angles.append(np.deg2rad(res['center_deg']))
                            values.append(rv)
                if angles:
                    angles.append(angles[0])
                    values.append(values[0])
                    ax.plot(angles, values, 'o-', linewidth=2.5, markersize=8,
                            color=colors[i], label=f'{rp}-year', alpha=0.8)
                    ax.fill(angles, values, alpha=0.15, color=colors[i])
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_title(title or 'Return Values by Direction',
                         fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            if n_periods == 1:
                figsize = figsize or (8, 8)
                fig = plt.figure(figsize=figsize)
                axes = [fig.add_subplot(111, projection='polar')]
            elif n_periods == 2:
                figsize = figsize or (14, 6)
                fig, axes = plt.subplots(1, 2, figsize=figsize,
                                         subplot_kw=dict(projection='polar'))
                axes = axes.flatten()
            else:
                figsize = figsize or (14, 10)
                fig, axes = plt.subplots(2, 2, figsize=figsize,
                                         subplot_kw=dict(projection='polar'))
                axes = axes.flatten()

            for i, rp in enumerate(return_periods):
                ax = axes[i]
                angles, values = [], []
                for sn in sector_names:
                    res = directional_results[sn]
                    if res.get('success') and 'return_values' in res:
                        rv = res['return_values'].get(rp)
                        if rv is not None and not np.isnan(rv):
                            angles.append(np.deg2rad(res['center_deg']))
                            values.append(rv)
                if not angles:
                    ax.text(0.5, 0.5, 'No data available',
                            ha='center', va='center', transform=ax.transAxes)
                    continue
                angles.append(angles[0])
                values.append(values[0])
                ax.plot(angles, values, 'o-', linewidth=2.5, markersize=8,
                        color=colors[i], alpha=0.8)
                ax.fill(angles, values, alpha=0.25, color=colors[i])
                ax.set_theta_zero_location('N')
                ax.set_theta_direction(-1)
                ax.set_title(f'{rp}-Year Return Value', fontsize=12,
                             fontweight='bold', pad=20)
                ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
                ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
                ax.grid(True, alpha=0.3)
                if show_values:
                    for angle, value in zip(angles[:-1], values[:-1]):
                        if not np.isnan(value):
                            ax.text(angle, value * 1.1, f'{value:.1f}',
                                    ha='center', va='center', fontsize=8,
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white', alpha=0.8))
            fig.suptitle(title or 'Return Values by Direction (m/s)',
                         fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()
        return fig, axes

    def __repr__(self) -> str:
        dist = f", distribution={self.distribution_name}" if self.distribution_name else ""
        return f"ExtremesAnalyzer(n_points={len(self.data)}, time_span={self.time_span:.1f} {self.time_unit}{dist})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_eva_family(method: str, distribution: str):
    """Raise ValueError if *distribution* is not in EVA_FAMILIES[method]."""
    allowed = EVA_FAMILIES.get(method, [])
    if distribution.lower() not in allowed:
        raise ValueError(
            f"Distribution {distribution!r} is not valid for method {method!r}. "
            f"Allowed: {allowed}. "
            f"Use one of the theoretically justified families for extreme value analysis."
        )
