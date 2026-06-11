"""
Data processor for statistical analysis
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
import warnings


class DataProcessor:
    """
    Container for numerical data and entry point for statistical analysis.

    Loads and validates array-like data, providing access to distribution
    fitting (:meth:`fit`), auto-fitting (:meth:`get_auto_fitter`), extreme
    value analysis (:meth:`get_extremes_analyzer`), and Monte Carlo stability
    analysis via the underlying :class:`~magica.core.magic_adjuster.MagicAdjuster`.
    """

    def __init__(self, data: Union[np.ndarray, list, pd.Series, pd.DataFrame] = None):
        """
        Parameters
        ----------
        data : array-like, optional
            Numpy array, list, pandas Series, or DataFrame.
        """
        self.data = None
        self.metadata: Dict[str, Any] = {}
        self._original_data = None  # preserved for extremes datetime extraction

        if data is not None:
            self.load_data(data)

    def load_data(
        self, data: Union[np.ndarray, list, pd.Series, pd.DataFrame]
    ) -> 'DataProcessor':
        """
        Load and validate data, converting to a 1-D numpy array.

        NaN values are removed with a warning.

        Parameters
        ----------
        data : array-like

        Returns
        -------
        DataProcessor
        """
        self._original_data = data

        if isinstance(data, np.ndarray):
            self.data = data.flatten() if data.ndim > 1 else data
        elif isinstance(data, list):
            self.data = np.array(data, dtype=float)
        elif isinstance(data, pd.Series):
            self.data = data.values
        elif isinstance(data, pd.DataFrame):
            self.data = data.values.flatten()
        else:
            try:
                self.data = np.array(data, dtype=float).flatten()
            except (ValueError, TypeError):
                raise TypeError(f"Cannot convert type {type(data)} to numpy array.")

        if np.any(np.isnan(self.data)):
            n_nan = int(np.sum(np.isnan(self.data)))
            warnings.warn(f"Found {n_nan} NaN values. They will be removed.")
            self.data = self.data[~np.isnan(self.data)]

        if len(self.data) == 0:
            raise ValueError("No valid data points after removing NaN values.")

        self._update_metadata()
        return self

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_data_array(self) -> np.ndarray:
        """Return the internal data array (no copy — same object)."""
        if self.data is None:
            raise ValueError("No data has been loaded.")
        return self.data

    def get_basic_stats(self) -> Dict[str, Any]:
        """Return basic descriptive statistics."""
        if self.data is None:
            raise ValueError("No data has been loaded.")
        return {
            'count': len(self.data),
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data, ddof=1)),
            'var': float(np.var(self.data, ddof=1)),
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data)),
            'median': float(np.median(self.data)),
            'q25': float(np.percentile(self.data, 25)),
            'q75': float(np.percentile(self.data, 75)),
        }

    # ------------------------------------------------------------------
    # Internal adjuster factory (kept for MagicAdjuster / MC internal use)
    # ------------------------------------------------------------------

    def _get_adjuster(self):
        """Create a :class:`~magica.core.magic_adjuster.MagicAdjuster` for this data."""
        from .magic_adjuster import MagicAdjuster
        return MagicAdjuster(self)

    # ------------------------------------------------------------------
    # Distribution fitting — primary API
    # ------------------------------------------------------------------

    def fit(self, distribution: Union[str, object], **kwargs):
        """
        Fit a distribution and return an immutable :class:`~magica.core.magic_adjuster.FitResult`.

        Each call is independent; multiple fits on the same processor coexist
        without state collision.

        Parameters
        ----------
        distribution : str or scipy continuous distribution
            e.g. ``'weibull'``, ``'gamma'``, ``scipy.stats.genextreme``, …
        **kwargs
            Passed to ``distribution.fit()`` (e.g. ``floc=0``).

        Returns
        -------
        FitResult

        Examples
        --------
        >>> import magica as ma, numpy as np
        >>> data = ma.read_data(np.random.weibull(2, 500) * 8)
        >>> fit_w = data.fit('weibull', floc=0)
        >>> fit_g = data.fit('gamma')
        >>> # Independent results:
        >>> fit_w.name, fit_g.name
        ('weibull', 'gamma')
        >>> fit_w.ppf(0.99)
        >>> fit_w.goodness_of_fit('ks')
        """
        return self._get_adjuster().fit_distribution(distribution, **kwargs)

    def fit_distribution(self, distribution: Union[str, object], **kwargs):
        """
        Alias for :meth:`fit` — returns a :class:`~magica.core.magic_adjuster.FitResult`.

        .. deprecated::
            The old behaviour (returning ``self`` and storing state) is removed.
            This method now behaves identically to :meth:`fit`.
        """
        return self.fit(distribution, **kwargs)

    # ------------------------------------------------------------------
    # Factory methods for higher-level tools
    # ------------------------------------------------------------------

    def get_auto_fitter(self, candidates=None, criterion='rmse'):
        """
        Create an :class:`~magica.core.auto_fitter.AutoFitter` for this data.

        Parameters
        ----------
        candidates : list of str, optional
        criterion : str, default ``'rmse'``

        Returns
        -------
        AutoFitter

        Examples
        --------
        >>> auto = data.get_auto_fitter()
        >>> best = auto.fit_best_distribution()
        >>> print(best.name, best.goodness_of_fit('rmse'))
        """
        from .auto_fitter import AutoFitter
        if self.data is None:
            raise ValueError("No data has been loaded.")
        return AutoFitter(self, candidates=candidates, criterion=criterion)

    def get_extremes_analyzer(
        self,
        times: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]] = None,
        time_unit: str = 'years',
    ):
        """
        Create an :class:`~magica.core.extremes_analyzer.ExtremesAnalyzer`.

        Parameters
        ----------
        times : array-like, optional
            Datetime or numeric times corresponding to the data.
        time_unit : str, default ``'years'``

        Returns
        -------
        ExtremesAnalyzer

        Examples
        --------
        >>> import pandas as pd
        >>> dates = pd.date_range('2000-01-01', periods=1000, freq='D')
        >>> series = pd.Series(values, index=dates)
        >>> processor = ma.read_data(series)
        >>> extremes = processor.get_extremes_analyzer()
        """
        from .extremes_analyzer import ExtremesAnalyzer
        if self.data is None:
            raise ValueError("No data has been loaded.")
        return ExtremesAnalyzer(self, times=times, time_unit=time_unit)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_metadata(self):
        if self.data is not None:
            self.metadata['length'] = len(self.data)
            self.metadata['dtype'] = str(self.data.dtype)
            self.metadata['last_updated'] = pd.Timestamp.now()

    def __repr__(self) -> str:
        if self.data is None:
            return "DataProcessor(no data loaded)"
        return f"DataProcessor(length={len(self.data)}, dtype={self.data.dtype})"

    def __len__(self) -> int:
        return 0 if self.data is None else len(self.data)
