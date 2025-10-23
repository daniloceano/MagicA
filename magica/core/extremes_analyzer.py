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
        
        Common distributions for extremes:
        - 'genextreme' - Generalized Extreme Value (GEV)
        - 'gumbel_r' - Gumbel distribution (Type I extreme)
        - 'gumbel_l' - Gumbel left (minimum extremes)
        - 'weibull_min' - Weibull (minimum extremes)
        - 'weibull_max' - Weibull (maximum extremes)
        
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
            raise ValueError("Must fit a distribution before calculating return values")
        
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
            raise ValueError("Must fit a distribution before calculating return periods")
        
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
    
    def extract_block_maxima(
        self,
        block_size: str = 'A',
        method: str = 'max'
    ) -> Tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
        """
        Extract block maxima (or minima) from time series.
        
        This is commonly used for GEV analysis where annual maxima
        are extracted from the data.
        
        Parameters
        ----------
        block_size : str, default='A'
            Block size for resampling. Uses pandas offset aliases:
            - 'A' or 'Y': Annual
            - 'Q': Quarterly
            - 'M': Monthly
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
        >>> annual_max, times = extremes.extract_block_maxima(block_size='A')
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
        min_separation: Optional[Union[str, pd.Timedelta]] = None
    ) -> Tuple[np.ndarray, Optional[Union[pd.DatetimeIndex, np.ndarray]]]:
        """
        Extract peaks over threshold (POT) from time series.
        
        This method is used for GPD (Generalized Pareto Distribution) analysis.
        
        Parameters
        ----------
        threshold : float
            Threshold value for peak detection
        min_separation : str or Timedelta, optional
            Minimum time separation between peaks to avoid clustering.
            Examples: '1D', '12H', pd.Timedelta(days=1)
            If None, all exceedances are returned.
            
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
        >>> # Extract peaks with minimum 1-day separation
        >>> peaks, times = extremes.peaks_over_threshold(
        ...     threshold=20.0,
        ...     min_separation='1D'
        ... )
        """
        # Find values exceeding threshold
        exceed_mask = self.data > threshold
        exceedances = self.data[exceed_mask]
        exceed_times = self.times[exceed_mask] if self.times is not None else None
        
        if min_separation is None or exceed_times is None:
            return exceedances, exceed_times
        
        # Decluster - keep only peaks separated by min_separation
        if not self.has_datetime:
            warnings.warn(
                "min_separation requires datetime information. "
                "Returning all exceedances without declustering."
            )
            return exceedances, exceed_times
        
        if isinstance(min_separation, str):
            min_separation = pd.Timedelta(min_separation)
        
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
            - data_length: Number of data points
            - time_span: Total time span in time_unit units
            - max_value: Maximum value in dataset
            - min_value: Minimum value in dataset
            - mean_value: Mean value
            - has_datetime: Whether datetime information is available
            - distribution: Fitted distribution name (if fitted)
        """
        summary = {
            'data_length': len(self.data),
            'time_span': self.time_span,
            'time_unit': self.time_unit,
            'max_value': float(np.max(self.data)),
            'min_value': float(np.min(self.data)),
            'mean_value': float(np.mean(self.data)),
            'has_datetime': self.has_datetime,
            'distribution': self.distribution_name
        }
        
        if self.has_datetime:
            summary['start_date'] = str(self.times[0])
            summary['end_date'] = str(self.times[-1])
        
        if self.fitted_params is not None:
            summary['fitted_parameters'] = self.fitted_params
            
        return summary
    
    def plot_return_levels(
        self,
        return_periods: Optional[np.ndarray] = None,
        empirical: bool = True,
        confidence_level: Optional[float] = None
    ):
        """
        Plot return level plot (return value vs return period).
        
        Parameters
        ----------
        return_periods : array-like, optional
            Return periods to plot. If None, uses logarithmic spacing.
        empirical : bool, default=True
            Whether to include empirical return values
        confidence_level : float, optional
            Confidence level for confidence intervals (e.g., 0.95)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        ax : matplotlib.axes.Axes
            Axes object
        """
        import matplotlib.pyplot as plt
        
        if self._adjuster is None or self.fitted_params is None:
            raise ValueError("Must fit a distribution before plotting return levels")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
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
        ax.set_title('Return Level Plot')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
        
        plt.tight_layout()
        return fig, ax
    
    def __repr__(self) -> str:
        """String representation of the analyzer."""
        dist_info = f", distribution={self.distribution_name}" if self.distribution_name else ""
        time_info = f", time_span={self.time_span:.1f} {self.time_unit}"
        return f"ExtremesAnalyzer(n_points={len(self.data)}{time_info}{dist_info})"
