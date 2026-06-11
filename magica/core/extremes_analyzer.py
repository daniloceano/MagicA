"""
Extreme values analysis for statistical data

This module provides tools for analyzing extreme values in time series data,
including return period and return value analysis.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Union, Optional, Dict, Any, Tuple
from scipy import stats

from .data_processor import DataProcessor


class ExtremesAnalyzer:
    """
    Extreme values analysis with return period and return value calculations.
    
    This class analyzes extreme values in time series data, calculating return
    periods and return values using statistical distributions fitted to the data.
    
    The analyzer supports multiple input formats:
    - Pandas Series with datetime index
    - Pandas DataFrame with time column
    - Paired numpy arrays (times, values)
    - Simple numpy array (assumes uniform time spacing)
    
    Parameters
    ----------
    data_processor : DataProcessor
        Processor instance with loaded data
    times : array-like, optional
        Time values corresponding to data points. Can be:
        - datetime array/Series
        - numeric array (e.g., years, days)
        - None if data is pandas Series with datetime index
    time_unit : str, default='years'
        Time unit for return period calculations ('years', 'days', 'hours', 'months')
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import magica as ma
    >>> 
    >>> # Using pandas Series with datetime index
    >>> dates = pd.date_range('2000-01-01', periods=1000, freq='D')
    >>> values = np.random.weibull(2, 1000) * 10
    >>> series = pd.Series(values, index=dates)
    >>> 
    >>> processor = ma.read_data(series)
    >>> extremes = processor.get_extremes_analyzer()
    >>> extremes.fit_distribution('genextreme')
    >>> 
    >>> # Calculate 100-year return value
    >>> rv_100 = extremes.return_value(100)
    >>> print(f"100-year return value: {rv_100:.2f}")
    """
    
    def __init__(
        self,
        data_processor: DataProcessor,
        times: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]] = None,
        time_unit: str = 'years'
    ):
        """
        Initialize ExtremesAnalyzer with data and time information.
        
        Parameters
        ----------
        data_processor : DataProcessor
            The data processor containing values to analyze
        times : array-like, optional
            Time values or datetime index
        time_unit : str, default='years'
            Unit for return period calculations
        """
        if data_processor.data is None:
            raise ValueError("DataProcessor must contain data before extreme analysis")
        
        self.data_processor = data_processor
        self.data = data_processor.get_data_array()
        self.time_unit = time_unit
        
        # Handle different time input formats
        self._process_times(times)
        
        # Internal adjuster for distribution fitting
        self._adjuster = None
        self.distribution_name = None
        self.fitted_params = None
        
    def _process_times(self, times: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]]):
        """
        Process time information from various input formats.
        
        Parameters
        ----------
        times : array-like or None
            Time information in various formats
        """
        if times is None:
            # Check if original data was pandas Series with datetime index
            if hasattr(self.data_processor, '_original_data'):
                original = self.data_processor._original_data
                if isinstance(original, pd.Series) and isinstance(original.index, pd.DatetimeIndex):
                    times = original.index
                    
        if times is None:
            # No time information - assume uniform spacing
            warnings.warn(
                "No time information provided. Assuming uniform time spacing. "
                "Return periods will be in units of observation count."
            )
            self.times = np.arange(len(self.data))
            self.has_datetime = False
            self.time_span = len(self.data)
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
                self.time_span = np.ptp(self.times)  # max - min
        else:
            # Numpy array or list
            times_array = np.array(times)
            if np.issubdtype(times_array.dtype, np.datetime64):
                self.times = pd.DatetimeIndex(times_array)
                self.has_datetime = True
                self.time_span = self._calculate_time_span(self.times)
            else:
                self.times = times_array
                self.has_datetime = False
                self.time_span = np.ptp(self.times) if len(times_array) > 1 else len(times_array)
                
    def _calculate_time_span(self, times: pd.DatetimeIndex) -> float:
        """
        Calculate time span in specified units from datetime index.
        
        Parameters
        ----------
        times : pd.DatetimeIndex
            Datetime index
            
        Returns
        -------
        float
            Time span in specified units
        """
        delta = times[-1] - times[0]
        
        if self.time_unit == 'years':
            return delta.total_seconds() / (365.25 * 24 * 3600)
        elif self.time_unit == 'days':
            return delta.total_seconds() / (24 * 3600)
        elif self.time_unit == 'hours':
            return delta.total_seconds() / 3600
        elif self.time_unit == 'months':
            return delta.total_seconds() / (30.44 * 24 * 3600)  # Average month
        else:
            raise ValueError(f"Unknown time unit: {self.time_unit}")
    
    def _get_adjuster(self):
        """Get or create the internal adjuster."""
        if self._adjuster is None:
            from .magic_adjuster import MagicAdjuster
            self._adjuster = MagicAdjuster(self.data_processor)
        return self._adjuster
    
    def fit_distribution(self, distribution: Union[str, object], **kwargs) -> 'ExtremesAnalyzer':
        """
        Fit a statistical distribution for extreme value analysis.
        
        This method can be called directly on ExtremesAnalyzer to fit
        distributions to the extreme data, without requiring a prior
        fit on the underlying DataProcessor.
        
        Common distributions for extremes:
        - 'genextreme' - Generalized Extreme Value (GEV)
        - 'gumbel_r' - Gumbel distribution (Type I extreme)
        - 'gumbel_l' - Gumbel left (minimum extremes)
        - 'weibull_min' - Weibull (minimum extremes)
        - 'weibull_max' - Weibull (maximum extremes)
        - 'genpareto' - Generalized Pareto Distribution (GPD, for POT)
        
        Parameters
        ----------
        distribution : str or scipy.stats distribution
            Distribution to fit
        **kwargs : dict
            Additional arguments passed to fit method
            
        Returns
        -------
        ExtremesAnalyzer
            Self for method chaining
            
        Examples
        --------
        >>> # Direct fitting on extremes analyzer
        >>> extremes = processor.get_extremes_analyzer()
        >>> annual_max, times = extremes.extract_block_maxima('Y')
        >>> 
        >>> # Create new processor with maxima and fit
        >>> maxima_proc = ma.read_data(annual_max)
        >>> maxima_extremes = maxima_proc.get_extremes_analyzer()
        >>> maxima_extremes.fit_distribution('genextreme')
        >>> rv_100 = maxima_extremes.return_value(100)
        """
        adjuster = self._get_adjuster()
        adjuster.fit_distribution(distribution, **kwargs)
        
        self.distribution_name = adjuster.distribution_name
        self.fitted_params = adjuster.fitted_params
        
        return self
    
    def return_value(self, return_period: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate return value for given return period(s).
        
        The return value is the value expected to be exceeded once every
        T time units on average, where T is the return period.
        
        Parameters
        ----------
        return_period : float or array-like
            Return period(s) in time_unit units (e.g., years)
            
        Returns
        -------
        float or ndarray
            Return value(s) corresponding to the return period(s)
            
        Examples
        --------
        >>> # Single return value
        >>> rv_100 = extremes.return_value(100)  # 100-year return value
        >>> 
        >>> # Multiple return values
        >>> periods = [10, 50, 100, 500]
        >>> rv = extremes.return_value(periods)
        """
        if self._adjuster is None or self.fitted_params is None:
            raise ValueError(
                "Must fit a distribution before calculating return values. "
                "Use extremes.fit_distribution('genextreme') or similar first."
            )
        
        # Convert return period to exceedance probability
        # P(X > x) = 1/T  =>  P(X <= x) = 1 - 1/T
        return_period = np.asarray(return_period)
        exceedance_prob = 1.0 / return_period
        non_exceedance_prob = 1.0 - exceedance_prob
        
        # Calculate quantile (return value) for non-exceedance probability
        return_values = self._adjuster.ppf(non_exceedance_prob)
        
        return return_values
    
    def return_period(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate return period for given value(s).
        
        The return period is the average time interval between exceedances
        of the given value.
        
        Parameters
        ----------
        value : float or array-like
            Value(s) for which to calculate return period
            
        Returns
        -------
        float or ndarray
            Return period(s) in time_unit units
            
        Examples
        --------
        >>> # Return period for specific value
        >>> rp = extremes.return_period(25.0)
        >>> print(f"A value of 25 has a return period of {rp:.1f} years")
        >>> 
        >>> # Return periods for multiple values
        >>> values = [20, 25, 30, 35]
        >>> rp = extremes.return_period(values)
        """
        if self._adjuster is None or self.fitted_params is None:
            raise ValueError(
                "Must fit a distribution before calculating return periods. "
                "Use extremes.fit_distribution('genextreme') or similar first."
            )
        
        # Calculate exceedance probability: P(X > value)
        value = np.asarray(value)
        exceedance_prob = 1.0 - self._adjuster.cdf(value)
        
        # Return period T = 1 / P(exceedance)
        # Avoid division by zero
        return_periods = np.where(
            exceedance_prob > 0,
            1.0 / exceedance_prob,
            np.inf
        )
        
        return return_periods
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Percent point function (inverse of CDF) of the fitted distribution.
        
        Parameters
        ----------
        q : float or array-like
            Probability value(s) between 0 and 1
            
        Returns
        -------
        float or ndarray
            Quantile(s) corresponding to the probability value(s)
        """
        if self._adjuster is None or self.fitted_params is None:
            raise ValueError(
                "Must fit a distribution before using ppf. "
                "Use extremes.fit_distribution('genextreme') or similar first."
            )
        return self._adjuster.ppf(q)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function of the fitted distribution.
        
        Parameters
        ----------
        x : float or array-like
            Value(s) at which to evaluate the CDF
            
        Returns
        -------
        float or ndarray
            Probability value(s) corresponding to x
        """
        if self._adjuster is None or self.fitted_params is None:
            raise ValueError(
                "Must fit a distribution before using cdf. "
                "Use extremes.fit_distribution('genextreme') or similar first."
            )
        return self._adjuster.cdf(x)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability density function of the fitted distribution.
        
        Parameters
        ----------
        x : float or array-like
            Value(s) at which to evaluate the PDF
            
        Returns
        -------
        float or ndarray
            Probability density value(s) corresponding to x
        """
        if self._adjuster is None or self.fitted_params is None:
            raise ValueError(
                "Must fit a distribution before using pdf. "
                "Use extremes.fit_distribution('genextreme') or similar first."
            )
        return self._adjuster.pdf(x)
    
    def goodness_of_fit(self, method: str, **kwargs):
        """
        Perform goodness-of-fit test for the fitted distribution.
        
        Parameters
        ----------
        method : str
            Test method: 'ks' (Kolmogorov-Smirnov), 'chi2' (Chi-square),
            'ad' (Anderson-Darling), or 'rmse' (Root Mean Square Error)
        **kwargs : dict
            Additional arguments passed to the goodness-of-fit method
            
        Returns
        -------
        dict or float
            Test results (dict for statistical tests, float for RMSE)
            
        Examples
        --------
        >>> extremes.fit_distribution('genextreme')
        >>> ks_test = extremes.goodness_of_fit('ks')
        >>> print(f"KS statistic: {ks_test['ks_statistic']:.4f}")
        >>> print(f"P-value: {ks_test['p_value']:.4f}")
        """
        if self._adjuster is None or self.fitted_params is None:
            raise ValueError(
                "Must fit a distribution before performing goodness-of-fit test. "
                "Use extremes.fit_distribution('genextreme') or similar first."
            )
        return self._adjuster.goodness_of_fit(method, **kwargs)
    
    def extract_block_maxima(
        self,
        block_size: str = 'YE',
        method: str = 'max'
    ) -> Tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
        """
        Extract block maxima (or minima) from time series.
        
        This is commonly used for GEV analysis where annual maxima
        are extracted from the data.
        
        Parameters
        ----------
        block_size : str, default='YE'
            Block size for resampling. Uses pandas offset aliases:
            - 'YE' or 'Y': Annual
            - 'QE': Quarterly
            - 'ME': Monthly
            - 'W': Weekly
            - 'D': Daily
        method : str, default='max'
            Aggregation method: 'max' or 'min'
            
        Returns
        -------
        values : ndarray
            Block maxima/minima values
        times : DatetimeIndex or None
            Block center times (if datetime available)
            
        Examples
        --------
        >>> # Extract annual maxima
        >>> annual_max, times = extremes.extract_block_maxima(block_size='YE')
        >>> 
        >>> # Extract monthly minima
        >>> monthly_min, times = extremes.extract_block_maxima(block_size='M', method='min')
        """
        if not self.has_datetime:
            raise ValueError(
                "Block maxima extraction requires datetime information. "
                "Provide times as datetime array when creating ExtremesAnalyzer."
            )
        
        # Create pandas Series with datetime index
        series = pd.Series(self.data, index=self.times)
        
        # Resample and aggregate
        if method == 'max':
            resampled = series.resample(block_size).max()
        elif method == 'min':
            resampled = series.resample(block_size).min()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'max' or 'min'")
        
        # Remove NaN values
        resampled = resampled.dropna()
        
        return resampled.values, resampled.index
    
    def peaks_over_threshold(
        self,
        threshold: float,
        min_separation: Optional[Union[str, pd.Timedelta, int, float]] = None,
        event_wise: bool = False
    ) -> Tuple[np.ndarray, Optional[Union[pd.DatetimeIndex, np.ndarray]]]:
        """
        Extract peaks over threshold (POT) from time series.
        
        This method is used for GPD (Generalized Pareto Distribution) analysis.
        
        Parameters
        ----------
        threshold : float
            Threshold value for peak detection
        min_separation : str, Timedelta, int, or float, optional
            Minimum time separation between peaks to avoid clustering.
            Can be:
            - String: '1D', '12H', etc. (pandas time string)
            - pd.Timedelta: pd.Timedelta(days=1)
            - Numeric: interpreted as days (e.g., 3 means 3 days)
            - 'event': Use event-wise declustering (requires event_wise=True)
            If None, all exceedances are returned.
        event_wise : bool, default=False
            If True, identifies consecutive periods above threshold as single events
            and returns only the maximum value from each event. This is useful for
            identifying storm events where multiple consecutive days exceed the threshold.
            When True, min_separation is ignored.
            
        Returns
        -------
        exceedances : ndarray
            Values exceeding the threshold
        times : DatetimeIndex or ndarray or None
            Times of exceedances
            
        Examples
        --------
        >>> # Extract all peaks over threshold
        >>> peaks, times = extremes.peaks_over_threshold(threshold=20.0)
        >>> 
        >>> # Extract peaks with minimum 1-day separation (numeric)
        >>> peaks, times = extremes.peaks_over_threshold(threshold=20.0, min_separation=1)
        >>> 
        >>> # Extract peaks with minimum 1-day separation (string)
        >>> peaks, times = extremes.peaks_over_threshold(
        ...     threshold=20.0,
        ...     min_separation='1D'
        ... )
        >>> 
        >>> # Extract one peak per consecutive event (storm-wise)
        >>> peaks, times = extremes.peaks_over_threshold(threshold=20.0, event_wise=True)
        """
        # Find values exceeding threshold
        exceed_mask = self.data > threshold
        
        if not self.has_datetime:
            exceedances = self.data[exceed_mask]
            exceed_times = None
            if min_separation is not None or event_wise:
                warnings.warn(
                    "min_separation and event_wise require datetime information. "
                    "Returning all exceedances without declustering."
                )
            return exceedances, exceed_times
        
        # Event-wise declustering: identify consecutive periods as single events
        if event_wise:
            # Create pandas Series for easier manipulation
            series = pd.Series(self.data, index=self.times)
            
            # Identify groups of consecutive exceedances
            # Create a group ID that increments when there's a gap
            exceed_series = series[exceed_mask]
            
            if len(exceed_series) == 0:
                return np.array([]), pd.DatetimeIndex([])
            
            # Calculate time differences between consecutive exceedances
            time_diffs = exceed_series.index.to_series().diff()
            
            # Expected time step (most common difference in original series)
            original_diffs = pd.Series(self.times).diff().dropna()
            expected_step = original_diffs.mode()[0] if len(original_diffs) > 0 else pd.Timedelta(days=1)
            
            # A new event starts when the gap is larger than expected step
            # (i.e., there's at least one day below threshold)
            is_new_event = time_diffs > expected_step
            event_ids = is_new_event.cumsum()
            
            # For each event, get the maximum value and its time
            event_maxima = []
            event_times = []
            
            for event_id in event_ids.unique():
                event_data = exceed_series[event_ids == event_id]
                max_idx = event_data.idxmax()
                event_maxima.append(event_data[max_idx])
                event_times.append(max_idx)
            
            return np.array(event_maxima), pd.DatetimeIndex(event_times)
        
        # Regular declustering with min_separation
        exceedances = self.data[exceed_mask]
        exceed_times = self.times[exceed_mask]
        
        if min_separation is None:
            return exceedances, exceed_times
        
        # Convert min_separation to Timedelta
        if isinstance(min_separation, str):
            min_separation = pd.Timedelta(min_separation)
        elif isinstance(min_separation, (int, float)):
            # Interpret numeric values as days
            min_separation = pd.Timedelta(days=min_separation)
        
        # Decluster algorithm
        keep_indices = [0]  # Always keep first peak
        for i in range(1, len(exceed_times)):
            time_diff = exceed_times[i] - exceed_times[keep_indices[-1]]
            if time_diff >= min_separation:
                keep_indices.append(i)
        
        declustered_values = exceedances[keep_indices]
        declustered_times = exceed_times[keep_indices]
        
        return declustered_values, declustered_times
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for extreme value analysis.
        
        Returns
        -------
        dict
            Dictionary with summary statistics including:
            - n_observations: Number of data points
            - time_span: Total time span in time_unit units
            - time_unit: Time unit used for calculations
            - max: Maximum value in dataset
            - min: Minimum value in dataset
            - mean: Mean value
            - std: Standard deviation
            - percentile_95: 95th percentile
            - percentile_99: 99th percentile
            - has_datetime: Whether datetime information is available
            - distribution_name: Fitted distribution name (if fitted)
            - distribution_params: Fitted parameters (if fitted)
        """
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
            'distribution_name': self.distribution_name
        }
        
        if self.has_datetime:
            summary['start_date'] = str(self.times[0])
            summary['end_date'] = str(self.times[-1])
        
        if self.fitted_params is not None:
            summary['distribution_params'] = self.fitted_params
            
        return summary
    
    def plot_return_levels(
        self,
        ax=None,
        return_periods: Optional[np.ndarray] = None,
        empirical: bool = True,
        confidence_level: Optional[float] = None,
        title: Optional[str] = None
    ):
        """
        Plot return level plot (return value vs return period).
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure and axes.
        return_periods : array-like, optional
            Return periods to plot. If None, uses logarithmic spacing.
        empirical : bool, default=True
            Whether to include empirical return values
        confidence_level : float, optional
            Confidence level for confidence intervals (e.g., 0.95)
        title : str, optional
            Plot title. If None, uses default title.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object (None if ax was provided)
        ax : matplotlib.axes.Axes
            Axes object
            
        Examples
        --------
        >>> # Create standalone plot
        >>> fig, ax = extremes.plot_return_levels()
        >>> 
        >>> # Plot on existing axes
        >>> fig, ax = plt.subplots()
        >>> extremes.plot_return_levels(ax=ax, title='Custom Title')
        """
        import matplotlib.pyplot as plt
        
        if self._adjuster is None or self.fitted_params is None:
            raise ValueError("Must fit a distribution before plotting return levels")
        
        # Create figure if ax not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = None
        
        # Default return periods
        if return_periods is None:
            return_periods = np.logspace(0, 3, 50)  # 1 to 1000
        
        # Calculate theoretical return values
        theoretical_rv = self.return_value(return_periods)
        
        # Plot theoretical curve
        ax.plot(return_periods, theoretical_rv, 'r-', linewidth=2, 
                label=f'Theoretical ({self.distribution_name})')
        
        # Plot empirical points if requested
        if empirical:
            # Sort data and calculate empirical return periods
            sorted_data = np.sort(self.data)[::-1]  # Descending order
            n = len(sorted_data)
            empirical_rp = (n + 1) / np.arange(1, n + 1)
            
            ax.plot(empirical_rp, sorted_data, 'bo', alpha=0.6, 
                   markersize=4, label='Empirical')
        
        ax.set_xlabel(f'Return Period ({self.time_unit})')
        ax.set_ylabel('Return Value')
        ax.set_title(title if title else 'Return Level Plot')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
        
        if fig is not None:
            plt.tight_layout()
        
        return fig, ax
    
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
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Find optimal POT threshold ensuring sufficient independent samples.
        
        This function systematically searches for a threshold that provides
        at least `min_samples` independent exceedances after declustering.
        The search can prioritize varying percentiles or time separation first.
        
        Parameters
        ----------
        min_samples : int, default=50
            Minimum number of independent exceedances required
        percentile_min : float, default=90
            Minimum percentile to test (lower bound)
        percentile_max : float, default=99
            Maximum percentile to test (upper bound, starting point)
        percentile_step : float, default=1.0
            Step size for decreasing percentile
        min_separation_hours : float, default=48
            Minimum time separation between peaks in hours (starting point)
        max_separation_hours : float, default=120
            Maximum time separation to test in hours
        separation_step_hours : float, default=24
            Step size for increasing separation in hours
        vary_first : str, default='percentile'
            Which parameter to vary first: 'percentile' or 'separation'
            - 'percentile': Varies percentile from max to min, then increases separation
            - 'separation': Varies separation from min to max, then decreases percentile
        max_iterations : int, default=200
            Maximum total search iterations to prevent infinite loops
        verbose : bool, default=False
            Print progress information during search
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'threshold': Selected threshold value (float)
            - 'percentile': Percentile of threshold (float)
            - 'separation_hours': Time separation used in hours (float)
            - 'n_raw_exceedances': Total exceedances before declustering (int)
            - 'n_independent': Independent exceedances after declustering (int)
            - 'exceedances': Array of independent exceedance values (ndarray)
            - 'exceedance_times': Times of independent exceedances (DatetimeIndex or ndarray)
            - 'success': Whether minimum samples achieved (bool)
            - 'iterations': Number of iterations performed (int)
            - 'warning': Warning message if any (str, optional)
            
        Examples
        --------
        >>> # Default search (vary percentile first, 99->90, then increase separation)
        >>> result = extremes.find_optimal_pot_threshold(min_samples=50)
        >>> print(f"Threshold: {result['threshold']:.2f}")
        >>> print(f"Independent samples: {result['n_independent']}")
        >>> 
        >>> # Vary separation first, then percentile
        >>> result = extremes.find_optimal_pot_threshold(
        ...     min_samples=50,
        ...     vary_first='separation'
        ... )
        >>> 
        >>> # Custom percentile and separation ranges
        >>> result = extremes.find_optimal_pot_threshold(
        ...     min_samples=30,
        ...     percentile_min=85,
        ...     percentile_max=98,
        ...     min_separation_hours=24,
        ...     max_separation_hours=96,
        ...     verbose=True
        ... )
        
        Notes
        -----
        **Search Strategy:**
        
        When `vary_first='percentile'` (default):
        1. Start with min_separation_hours and percentile_max
        2. Decrease percentile by percentile_step until percentile_min
        3. If not enough samples, increase separation by separation_step_hours
        4. Repeat percentile search with new separation
        5. Continue until max_separation_hours or min_samples achieved
        
        When `vary_first='separation'`:
        1. Start with percentile_max and min_separation_hours
        2. Increase separation by separation_step_hours until max_separation_hours
        3. If not enough samples, decrease percentile by percentile_step
        4. Repeat separation search with new percentile
        5. Continue until percentile_min or min_samples achieved
        
        **Best Practices:**
        - For synoptic events (storms): min_separation_hours >= 48
        - For sub-daily events: min_separation_hours >= 12
        - Typical percentile range: 90-99 for extremes
        - Ensure min_samples >= 30 for reliable fitting (50+ recommended)
        """
        if not self.has_datetime:
            raise ValueError(
                "POT threshold search requires datetime information. "
                "Provide times when creating ExtremesAnalyzer."
            )
        
        if vary_first not in ['percentile', 'separation']:
            raise ValueError("vary_first must be 'percentile' or 'separation'")
        
        # Validate ranges
        if percentile_min >= percentile_max:
            raise ValueError("percentile_min must be < percentile_max")
        if min_separation_hours >= max_separation_hours:
            raise ValueError("min_separation_hours must be < max_separation_hours")
        
        # Initialize search variables
        iteration = 0
        best_result = None
        
        if vary_first == 'percentile':
            # Strategy: Vary percentile first, then increase separation
            separations = np.arange(min_separation_hours, max_separation_hours + 1, 
                                   separation_step_hours)
            
            for separation_hours in separations:
                if verbose:
                    print(f"\n--- Testing separation: {separation_hours}h ---")
                
                # Test percentiles from max to min
                current_percentile = percentile_max
                
                while current_percentile >= percentile_min:
                    iteration += 1
                    
                    if iteration > max_iterations:
                        break
                    
                    # Calculate threshold
                    threshold = np.percentile(self.data, current_percentile)
                    
                    # Count raw exceedances
                    n_raw = (self.data > threshold).sum()
                    
                    if n_raw == 0:
                        current_percentile -= percentile_step
                        continue
                    
                    try:
                        # Extract POT with current separation
                        sep_days = separation_hours / 24
                        exceedances, exc_times = self.peaks_over_threshold(
                            threshold=threshold,
                            min_separation=sep_days
                        )
                        
                        n_independent = len(exceedances)
                        
                        # Store result
                        result = {
                            'threshold': threshold,
                            'percentile': current_percentile,
                            'separation_hours': separation_hours,
                            'n_raw_exceedances': n_raw,
                            'n_independent': n_independent,
                            'exceedances': exceedances,
                            'exceedance_times': exc_times,
                            'success': n_independent >= min_samples,
                            'iterations': iteration
                        }
                        
                        # Update best result
                        if best_result is None or n_independent > best_result['n_independent']:
                            best_result = result.copy()
                        
                        if verbose:
                            status = "✓" if result['success'] else "•"
                            print(f"{status} p={current_percentile:.1f}, "
                                  f"thresh={threshold:.2f}, n={n_independent}")
                        
                        # Check if we found enough samples
                        if n_independent >= min_samples:
                            return result
                        
                    except Exception as e:
                        if verbose:
                            print(f"✗ p={current_percentile:.1f}: {str(e)}")
                    
                    # Decrease percentile
                    current_percentile -= percentile_step
                
                if iteration > max_iterations:
                    break
        
        else:  # vary_first == 'separation'
            # Strategy: Vary separation first, then decrease percentile
            percentiles = np.arange(percentile_max, percentile_min - percentile_step, 
                                   -percentile_step)
            
            for percentile in percentiles:
                if verbose:
                    print(f"\n--- Testing percentile: {percentile:.1f} ---")
                
                # Test separations from min to max
                current_separation = min_separation_hours
                
                while current_separation <= max_separation_hours:
                    iteration += 1
                    
                    if iteration > max_iterations:
                        break
                    
                    # Calculate threshold
                    threshold = np.percentile(self.data, percentile)
                    
                    # Count raw exceedances
                    n_raw = (self.data > threshold).sum()
                    
                    if n_raw == 0:
                        break  # No exceedances at this percentile
                    
                    try:
                        # Extract POT with current separation
                        sep_days = current_separation / 24
                        exceedances, exc_times = self.peaks_over_threshold(
                            threshold=threshold,
                            min_separation=sep_days
                        )
                        
                        n_independent = len(exceedances)
                        
                        # Store result
                        result = {
                            'threshold': threshold,
                            'percentile': percentile,
                            'separation_hours': current_separation,
                            'n_raw_exceedances': n_raw,
                            'n_independent': n_independent,
                            'exceedances': exceedances,
                            'exceedance_times': exc_times,
                            'success': n_independent >= min_samples,
                            'iterations': iteration
                        }
                        
                        # Update best result
                        if best_result is None or n_independent > best_result['n_independent']:
                            best_result = result.copy()
                        
                        if verbose:
                            status = "✓" if result['success'] else "•"
                            print(f"{status} sep={current_separation}h, "
                                  f"thresh={threshold:.2f}, n={n_independent}")
                        
                        # Check if we found enough samples
                        if n_independent >= min_samples:
                            return result
                        
                    except Exception as e:
                        if verbose:
                            print(f"✗ sep={current_separation}h: {str(e)}")
                    
                    # Increase separation
                    current_separation += separation_step_hours
                
                if iteration > max_iterations:
                    break
        
        # No solution found meeting min_samples requirement
        if best_result is None:
            # Return empty result
            return {
                'threshold': np.nan,
                'percentile': np.nan,
                'separation_hours': np.nan,
                'n_raw_exceedances': 0,
                'n_independent': 0,
                'exceedances': np.array([]),
                'exceedance_times': pd.DatetimeIndex([]) if self.has_datetime else None,
                'success': False,
                'iterations': iteration,
                'warning': 'No exceedances found in search range'
            }
        
        # Return best effort result
        best_result['warning'] = (
            f"Could not find threshold with {min_samples} samples. "
            f"Best result: {best_result['n_independent']} samples. "
            f"Consider relaxing constraints."
        )
        
        if verbose:
            print(f"\n⚠️  {best_result['warning']}")
        
        return best_result
    
    @staticmethod
    def plot_directional_return_values(
        directional_results: Dict[str, Dict[str, Any]],
        return_periods: list = None,
        colors: list = None,
        overlay: bool = False,
        show_values: bool = True,
        figsize: tuple = None,
        title: str = None
    ) -> tuple:
        """
        Create polar plots of return values by direction.
        
        This method creates polar plots showing how return values vary by wind
        direction for different return periods. Can create separate subplots for
        each return period or overlay all periods in a single plot.
        
        Parameters
        ----------
        directional_results : dict
            Dictionary with sector names as keys and result dictionaries as values.
            Each result dictionary must contain:
            - 'center_deg': Center angle of sector in degrees
            - 'return_values': Dictionary mapping return periods to values
            - 'success': Boolean indicating if fit was successful
        return_periods : list of int, optional
            Return periods to plot (max 4). If None, uses [10, 20, 50, 100].
            Examples: [10, 50], [1, 5, 10, 20], [100]
        colors : list of str, optional
            Colors for each return period. If None, uses default palette.
            Must have same length as return_periods.
        overlay : bool, default=False
            If True, plot all return periods on single polar plot.
            If False, create separate subplot for each return period.
        show_values : bool, default=True
            If True, display numeric values on the plot.
        figsize : tuple, optional
            Figure size (width, height). If None, automatically determined.
        title : str, optional
            Main figure title. If None, uses default title.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        axes : matplotlib.axes.Axes or array of Axes
            The axes object(s)
        
        Raises
        ------
        ValueError
            If return_periods has more than 4 values
            If colors length doesn't match return_periods length
            If directional_results is empty or invalid
        
        Examples
        --------
        >>> # Assuming you have fitted distributions for each sector
        >>> directional_results = {}
        >>> for sector in sectors:
        ...     # ... fit distribution and calculate return values ...
        ...     directional_results[sector] = {
        ...         'center_deg': center_angle,
        ...         'return_values': {10: rv10, 50: rv50, 100: rv100},
        ...         'success': True
        ...     }
        >>> 
        >>> # Create separate subplots for each return period
        >>> fig, axes = extremes.plot_directional_return_values(
        ...     directional_results,
        ...     return_periods=[10, 50, 100]
        ... )
        >>> 
        >>> # Create single overlay plot
        >>> fig, ax = extremes.plot_directional_return_values(
        ...     directional_results,
        ...     return_periods=[10, 50, 100],
        ...     overlay=True
        ... )
        >>> 
        >>> # Customize colors and appearance
        >>> fig, axes = extremes.plot_directional_return_values(
        ...     directional_results,
        ...     return_periods=[20, 100],
        ...     colors=['#FF6B6B', '#4ECDC4'],
        ...     figsize=(12, 6)
        ... )
        >>> plt.show()
        
        Notes
        -----
        - Maximum 4 return periods can be plotted
        - If return_periods <= 2 and not overlay, creates single row of subplots
        - If return_periods == 1 and not overlay, creates single subplot
        - In overlay mode, all periods shown on one polar plot with legend
        - Automatically handles missing/failed sectors
        """
        import matplotlib.pyplot as plt
        
        # Validate inputs
        if not directional_results:
            raise ValueError("directional_results cannot be empty")
        
        # Default return periods
        if return_periods is None:
            return_periods = [10, 20, 50, 100]
        
        # Validate return periods
        if len(return_periods) > 4:
            raise ValueError("Maximum 4 return periods can be plotted")
        if len(return_periods) == 0:
            raise ValueError("At least one return period must be specified")
        
        # Default colors
        if colors is None:
            default_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            colors = default_colors[:len(return_periods)]
        elif len(colors) != len(return_periods):
            raise ValueError("Number of colors must match number of return periods")
        
        # Extract sector information and sort by angle (not alphabetically)
        # This ensures the polar plot forms a proper circle
        sector_data = []
        for sector_name, result in directional_results.items():
            if 'center_deg' in result:
                sector_data.append((sector_name, result['center_deg']))
        
        # Sort by angle to get proper circular order
        sector_data.sort(key=lambda x: x[1])
        sector_names = [name for name, _ in sector_data]
        
        if not sector_names:
            raise ValueError("No sectors found in directional_results")
        
        # Determine subplot layout
        n_periods = len(return_periods)
        
        if overlay:
            # Single plot with all periods overlaid
            if figsize is None:
                figsize = (10, 10)
            
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='polar')
            axes = ax
            
            # Plot each return period
            for i, rp in enumerate(return_periods):
                angles = []
                values = []
                
                for sector_name in sector_names:
                    result = directional_results[sector_name]
                    if result.get('success', False) and 'return_values' in result:
                        rv_dict = result['return_values']
                        if rp in rv_dict and not np.isnan(rv_dict[rp]):
                            angle_center = result['center_deg']
                            angles.append(np.deg2rad(angle_center))
                            values.append(rv_dict[rp])
                
                if angles:
                    # Close the plot
                    angles.append(angles[0])
                    values.append(values[0])
                    
                    # Plot line
                    ax.plot(angles, values, 'o-', linewidth=2.5, markersize=8,
                           color=colors[i], label=f'{rp}-year', alpha=0.8)
                    ax.fill(angles, values, alpha=0.15, color=colors[i])
            
            # Format
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            if title is None:
                title = 'Return Values by Direction'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add value labels if requested
            if show_values:
                for i, rp in enumerate(return_periods):
                    angles = []
                    values = []
                    
                    for sector_name in sector_names:
                        result = directional_results[sector_name]
                        if result.get('success', False) and 'return_values' in result:
                            rv_dict = result['return_values']
                            if rp in rv_dict and not np.isnan(rv_dict[rp]):
                                angle_center = result['center_deg']
                                angles.append(np.deg2rad(angle_center))
                                values.append(rv_dict[rp])
                    
                    if angles and i == 0:  # Only show values for first period to avoid clutter
                        # Create text box per sector with all values
                        for j, (angle, _) in enumerate(zip(angles, values)):
                            sector_name = sector_names[j]
                            result = directional_results[sector_name]
                            rv_dict = result.get('return_values', {})
                            
                            # Build text with all return periods
                            text_lines = []
                            for rp_val in return_periods:
                                if rp_val in rv_dict and not np.isnan(rv_dict[rp_val]):
                                    text_lines.append(f'{rp_val}yr: {rv_dict[rp_val]:.1f}')
                            
                            if text_lines:
                                text = '\n'.join(text_lines)
                                # Position text slightly outside the plot
                                max_val = max([v for v in values if not np.isnan(v)])
                                r_pos = max_val * 1.15
                                ax.text(angle, r_pos, text, ha='center', va='center',
                                       fontsize=7, bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white', alpha=0.8, edgecolor='gray'))
            
        else:
            # Separate subplots for each return period
            if n_periods == 1:
                # Single subplot
                if figsize is None:
                    figsize = (8, 8)
                fig = plt.figure(figsize=figsize)
                axes = [fig.add_subplot(111, projection='polar')]
                nrows, ncols = 1, 1
            elif n_periods == 2:
                # Single row
                if figsize is None:
                    figsize = (14, 6)
                fig, axes = plt.subplots(1, 2, figsize=figsize,
                                        subplot_kw=dict(projection='polar'))
                axes = axes.flatten()
            else:
                # Grid layout (2 rows)
                if figsize is None:
                    figsize = (14, 10)
                nrows = 2
                ncols = 2
                fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                        subplot_kw=dict(projection='polar'))
                axes = axes.flatten()
            
            # Plot each return period
            for i, rp in enumerate(return_periods):
                ax = axes[i]
                
                # Get return values for this period
                angles = []
                values = []
                
                for sector_name in sector_names:
                    result = directional_results[sector_name]
                    if result.get('success', False) and 'return_values' in result:
                        rv_dict = result['return_values']
                        if rp in rv_dict and not np.isnan(rv_dict[rp]):
                            angle_center = result['center_deg']
                            angles.append(np.deg2rad(angle_center))
                            values.append(rv_dict[rp])
                
                if not angles:
                    ax.text(0.5, 0.5, 'No data available',
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Close the plot
                angles.append(angles[0])
                values.append(values[0])
                
                # Plot
                ax.plot(angles, values, 'o-', linewidth=2.5, markersize=8,
                       color=colors[i], alpha=0.8)
                ax.fill(angles, values, alpha=0.25, color=colors[i])
                
                # Format
                ax.set_theta_zero_location('N')
                ax.set_theta_direction(-1)
                ax.set_title(f'{rp}-Year Return Value', fontsize=12,
                           fontweight='bold', pad=20)
                ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
                ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
                ax.grid(True, alpha=0.3)
                
                # Add value labels if requested
                if show_values:
                    for angle, value in zip(angles[:-1], values[:-1]):
                        if not np.isnan(value):
                            ax.text(angle, value * 1.1, f'{value:.1f}',
                                   ha='center', va='center', fontsize=8,
                                   bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white', alpha=0.8))
            
            # Set main title
            if title is None:
                title = 'Return Values by Direction (m/s)'
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        return fig, axes
    
    def __repr__(self) -> str:
        """String representation of the analyzer."""
        dist_info = f", distribution={self.distribution_name}" if self.distribution_name else ""
        time_info = f", time_span={self.time_span:.1f} {self.time_unit}"
        return f"ExtremesAnalyzer(n_points={len(self.data)}{time_info}{dist_info})"
