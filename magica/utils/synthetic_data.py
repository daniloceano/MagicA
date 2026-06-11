"""
Synthetic wind data generation for testing and demonstration

This module provides functions to generate realistic synthetic wind speed data
with temporal structure, storm events, and optionally directional characteristics.
Useful for testing extreme value analysis methods and creating reproducible examples.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, Union


def generate_wind_data(
    n_years: float = 30,
    freq: str = 'D',
    mean_wind: float = 8.0,
    weibull_shape: float = 2.5,
    seasonal_amplitude: float = 0.3,
    n_storms_per_year: float = 5,
    storm_duration_range: Tuple[int, int] = (2, 5),
    storm_intensity_range: Tuple[float, float] = (12, 20),
    start_date: str = '1990-01-01',
    random_seed: Optional[int] = None,
    create_plots: bool = False
) -> Union[pd.Series, Tuple[pd.Series, Dict[str, Union[Figure, Axes]]]]:
    """
    Generate synthetic wind speed data with realistic temporal structure.
    
    Creates wind speed time series with:
    - Weibull-distributed base wind speeds
    - Seasonal variation (stronger winds in winter)
    - Realistic storm events with multi-day persistence
    - Gaussian-shaped storm intensity profiles
    
    Parameters
    ----------
    n_years : float, default=30
        Number of years of data to generate
    freq : str, default='D'
        Frequency of observations. Pandas offset aliases:
        - 'D': Daily
        - 'h': Hourly
        - '6h': 6-hourly
        etc.
    mean_wind : float, default=8.0
        Mean wind speed (m/s) for base distribution
    weibull_shape : float, default=2.5
        Shape parameter for Weibull distribution
    seasonal_amplitude : float, default=0.3
        Amplitude of seasonal variation (as fraction of base wind)
        Higher values = stronger seasonal effect
    n_storms_per_year : float, default=5
        Average number of storm events per year
    storm_duration_range : tuple, default=(2, 5)
        (min, max) duration of storms in units of freq
        For daily data: (2, 5) means 2-5 day storms
        For hourly data: (12, 48) means 12-48 hour storms
    storm_intensity_range : tuple, default=(12, 20)
        (min, max) peak intensity to add during storms (m/s)
    start_date : str, default='1990-01-01'
        Start date for time series
    random_seed : int, optional
        Random seed for reproducibility
    create_plots : bool, default=False
        If True, create diagnostic plots
        
    Returns
    -------
    wind_series : pd.Series
        Time series of wind speeds with datetime index
    plots : dict, optional
        Dictionary with 'fig' and 'axes' if create_plots=True
        Contains time series and histogram plots
        
    Examples
    --------
    >>> # Generate 30 years of daily data
    >>> wind_data = generate_wind_data(n_years=30, random_seed=42)
    >>> print(wind_data.describe())
    >>> 
    >>> # Generate hourly data with plots
    >>> wind_data, plots = generate_wind_data(
    ...     n_years=10,
    ...     freq='h',
    ...     storm_duration_range=(12, 48),
    ...     create_plots=True
    ... )
    >>> plt.show()
    
    Notes
    -----
    The synthetic data generation follows these steps:
    1. Create datetime index based on frequency and duration
    2. Generate base Weibull-distributed wind speeds
    3. Apply sinusoidal seasonal variation (peak in winter)
    4. Add storm events with Gaussian intensity profiles
    5. Add small random variations for realism
    
    Storm events are placed randomly and have:
    - Random duration within specified range
    - Bell-shaped intensity (peaks in middle)
    - Persistence across multiple time steps
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate datetime index
    n_days = int(n_years * 365.25)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Adjust for frequency
    if freq != 'D':
        dates = pd.date_range(start=start_date, end=dates[-1], freq=freq)
    
    n_samples = len(dates)
    
    # Base wind speeds (Weibull distribution)
    base_wind = np.random.weibull(weibull_shape, n_samples) * mean_wind / 1.5 + mean_wind / 3
    
    # Add seasonal variation (stronger winds in winter, Northern Hemisphere)
    day_of_year = dates.dayofyear.values
    seasonal_factor = 1 + seasonal_amplitude * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    wind_speeds = base_wind * seasonal_factor
    
    # Add storm events
    n_storms = int(n_storms_per_year * n_years)
    storm_min_dur, storm_max_dur = storm_duration_range
    storm_min_int, storm_max_int = storm_intensity_range
    
    # Ensure room for longest possible storm
    storm_starts = np.random.choice(
        n_samples - storm_max_dur, 
        min(n_storms, n_samples - storm_max_dur), 
        replace=False
    )
    
    for storm_start in storm_starts:
        # Random duration and intensity
        storm_duration = np.random.randint(storm_min_dur, storm_max_dur + 1)
        peak_intensity = np.random.uniform(storm_min_int, storm_max_int)
        
        for step in range(storm_duration):
            if storm_start + step < n_samples:
                # Gaussian-shaped intensity profile
                relative_position = step / (storm_duration - 1) if storm_duration > 1 else 0.5
                intensity_factor = np.exp(-4 * (relative_position - 0.5)**2)
                
                # Add storm contribution
                wind_speeds[storm_start + step] += peak_intensity * intensity_factor
                wind_speeds[storm_start + step] += np.random.normal(0, 0.5)
    
    # Create pandas Series
    wind_series = pd.Series(wind_speeds, index=dates, name='Wind Speed (m/s)')
    
    if not create_plots:
        return wind_series

    import matplotlib.pyplot as plt  # lazy import
    # Create diagnostic plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Time series plot
    ax1.plot(dates, wind_speeds, linewidth=0.5, alpha=0.7, color='steelblue')
    ax1.axhline(wind_speeds.mean(), color='green', linestyle='--', 
                linewidth=1.5, label='Mean', alpha=0.7)
    ax1.axhline(np.percentile(wind_speeds, 95), color='orange', linestyle='--', 
                linewidth=1.5, label='95th percentile', alpha=0.7)
    ax1.axhline(np.percentile(wind_speeds, 99), color='red', linestyle='--', 
                linewidth=1.5, label='99th percentile', alpha=0.7)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Wind Speed (m/s)', fontsize=11)
    ax1.set_title(f'Synthetic Wind Speed Time Series ({n_years:.1f} Years)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(wind_speeds, bins=50, density=True, alpha=0.7, color='steelblue', 
             edgecolor='black', linewidth=0.5)
    ax2.axvline(wind_speeds.mean(), color='green', linestyle='--', 
                linewidth=2, label='Mean')
    ax2.axvline(np.percentile(wind_speeds, 95), color='orange', linestyle='--', 
                linewidth=2, label='95th percentile')
    ax2.axvline(np.percentile(wind_speeds, 99), color='red', linestyle='--', 
                linewidth=2, label='99th percentile')
    ax2.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax2.set_ylabel('Probability Density', fontsize=11)
    ax2.set_title('Wind Speed Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return wind_series, {'fig': fig, 'axes': (ax1, ax2)}


def generate_directional_wind_data(
    n_years: float = 10,
    freq: str = 'h',
    prevailing_direction: float = 270,
    prevailing_concentration: float = 1.5,
    secondary_direction: Optional[float] = 45,
    secondary_frequency: float = 0.3,
    secondary_concentration: float = 2.0,
    directional_speed_factors: Optional[Dict[Tuple[float, float], float]] = None,
    mean_wind: float = 8.0,
    weibull_shape: float = 2.0,
    seasonal_amplitude: float = 0.25,
    n_storms_per_year: float = 5,
    storm_directions: Optional[list] = None,
    storm_duration_range: Tuple[int, int] = (12, 24),
    storm_intensity_range: Tuple[float, float] = (15, 25),
    start_date: str = '2010-01-01',
    random_seed: Optional[int] = None,
    create_plots: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Generate synthetic wind data with directional characteristics.
    
    Creates wind speed and direction time series with:
    - Von Mises-distributed wind directions (circular normal)
    - Primary and optional secondary prevailing directions
    - Directional dependency of wind speeds (fetch effects)
    - Realistic storm events with directional preference
    - Seasonal variation
    
    Parameters
    ----------
    n_years : float, default=10
        Number of years of data to generate
    freq : str, default='h'
        Frequency of observations (e.g., 'h' for hourly, 'D' for daily)
    prevailing_direction : float, default=270
        Primary prevailing wind direction in degrees (0-360)
        0/360 = North, 90 = East, 180 = South, 270 = West
    prevailing_concentration : float, default=1.5
        Concentration parameter for von Mises distribution (higher = more concentrated)
    secondary_direction : float, optional, default=45
        Secondary prevailing direction in degrees
        If None, no secondary direction
    secondary_frequency : float, default=0.3
        Fraction of time with winds from secondary direction
    secondary_concentration : float, default=2.0
        Concentration for secondary direction
    directional_speed_factors : dict, optional
        Dictionary mapping direction ranges to speed multiplication factors.
        Accepts three key formats for flexibility:
        
        1. **Tuple ranges**: {(min_deg, max_deg): factor}
           Example: {(225, 315): 1.4, (0, 90): 0.8}
        
        2. **Sector names** (8 sectors): {'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'}
           Example: {'W': 1.4, 'SW': 1.4, 'N': 0.8, 'NE': 0.8}
           Each sector spans 45° centered at standard compass points
        
        3. **Numeric centers**: {center_deg: factor}
           Example: {270: 1.4, 0: 0.8}
           Automatically creates ±22.5° range around center
        
        Handles wrap-around ranges correctly (e.g., N sector: 337.5° to 22.5°)
        If None, uses default SW/W enhancement and N/NE reduction
    mean_wind : float, default=8.0
        Base mean wind speed (m/s)
    weibull_shape : float, default=2.0
        Shape parameter for base wind distribution
    seasonal_amplitude : float, default=0.25
        Amplitude of seasonal variation
    n_storms_per_year : float, default=5
        Average number of storm events per year
    storm_directions : list, optional
        Preferred directions for storms (degrees)
        If None, uses prevailing_direction ± 20°
    storm_duration_range : tuple, default=(12, 24)
        (min, max) duration of storms in units of freq
    storm_intensity_range : tuple, default=(15, 25)
        (min, max) peak intensity to add during storms (m/s)
    start_date : str, default='2010-01-01'
        Start date for time series
    random_seed : int, optional
        Random seed for reproducibility
    create_plots : bool, default=False
        If True, create diagnostic plots (wind rose, box plots, etc.)
        
    Returns
    -------
    wind_data : pd.DataFrame
        DataFrame with columns:
        - 'datetime': Timestamp index
        - 'wind_speed': Wind speed (m/s)
        - 'wind_direction': Wind direction (degrees, 0-360)
    plots : dict, optional
        Dictionary with plot objects if create_plots=True
        Keys: 'fig', 'axes' containing matplotlib Figure and Axes
        
    Examples
    --------
    >>> # Generate 10 years of hourly directional wind data
    >>> wind_data = generate_directional_wind_data(n_years=10, random_seed=42)
    >>> print(wind_data.head())
    >>> 
    >>> # With custom directional speed factors and plots
    >>> # Using sector names (simpler than degree ranges)
    >>> wind_data, plots = generate_directional_wind_data(
    ...     n_years=5,
    ...     directional_speed_factors={
    ...         'W': 1.5,   # Strong west winds
    ...         'SW': 1.5,  # Strong southwest winds
    ...         'N': 0.7,   # Weak north winds
    ...         'NE': 0.7   # Weak northeast winds
    ...     },
    ...     create_plots=True
    ... )
    >>> plt.show()
    >>> 
    >>> # Or using traditional degree ranges
    >>> speed_factors = {
    ...     (225, 315): 1.5,  # Strong SW-W winds
    ...     (0, 90): 0.7      # Weak N-NE winds
    ... }
    >>> wind_data = generate_directional_wind_data(
    ...     n_years=5,
    ...     directional_speed_factors=speed_factors
    ... )
    
    Notes
    -----
    The directional data generation:
    1. Generates directions using von Mises distributions
    2. Applies directional speed modulation (fetch effects)
    3. Adds seasonal variations
    4. Inserts storm events with preferred directions
    5. Ensures realistic temporal persistence
    
    Directional speed factors simulate real-world effects like:
    - Fetch (water vs land)
    - Terrain blocking
    - Thermal effects
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate datetime index
    n_days = int(n_years * 365.25)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    if freq != 'D':
        dates = pd.date_range(start=start_date, end=dates[-1], freq=freq)
    
    n_samples = len(dates)
    
    # Generate wind directions using von Mises distribution
    direction_base = np.random.vonmises(
        np.radians(prevailing_direction), 
        prevailing_concentration, 
        n_samples
    )
    wind_direction = np.degrees(direction_base) % 360
    
    # Add secondary prevailing direction if specified
    if secondary_direction is not None:
        secondary_mask = np.random.random(n_samples) < secondary_frequency
        direction_secondary = np.random.vonmises(
            np.radians(secondary_direction),
            secondary_concentration,
            np.sum(secondary_mask)
        )
        wind_direction[secondary_mask] = np.degrees(direction_secondary) % 360
    
    # Generate base wind speeds
    base_wind = np.random.weibull(weibull_shape, n_samples) * mean_wind / 1.5 + mean_wind / 3
    
    # Apply directional speed factors
    # Accept multiple input formats for convenience:
    # - {(min_deg, max_deg): factor}
    # - {'N': factor, 'NE': factor, ...} (8-sector names)
    # - {center_deg: factor} (numeric center -> +/- half-sector)
    if directional_speed_factors is None:
        # Default: enhance SW/W, reduce N/NE
        directional_speed_factors = {
            (225, 315): 1.4,
            (0, 90): 0.8
        }

    # Helper: map sector name to degree range (8-sector, 45° bins centered at 0,45,...)
    sector_name_to_range = {
        'N': (337.5, 22.5),
        'NE': (22.5, 67.5),
        'E': (67.5, 112.5),
        'SE': (112.5, 157.5),
        'S': (157.5, 202.5),
        'SW': (202.5, 247.5),
        'W': (247.5, 292.5),
        'NW': (292.5, 337.5)
    }

    # Normalize user input into a list of (dir_min, dir_max, factor)
    factor_ranges = []
    for key, factor in directional_speed_factors.items():
        # Tuple/list range provided
        if isinstance(key, (tuple, list)) and len(key) == 2:
            dir_min, dir_max = float(key[0]), float(key[1])
        # String sector name provided
        elif isinstance(key, str):
            key_up = key.strip().upper()
            if key_up in sector_name_to_range:
                dir_min, dir_max = sector_name_to_range[key_up]
            else:
                raise ValueError(f"Unknown directional sector name: {key}")
        # Numeric center degree provided -> create +/- half-sector (22.5°)
        elif isinstance(key, (int, float, np.floating, np.integer)):
            center = float(key) % 360
            half = 22.5
            dir_min = (center - half) % 360
            dir_max = (center + half) % 360
        else:
            raise ValueError("directional_speed_factors keys must be (min,max) tuples, sector names, or numeric centers")

        factor_ranges.append((float(dir_min), float(dir_max), float(factor)))

    # Apply factors; support wrap-around ranges (dir_min > dir_max)
    for dir_min, dir_max, factor in factor_ranges:
        if dir_min <= dir_max:
            mask = (wind_direction >= dir_min) & (wind_direction <= dir_max)
        else:
            # Wrap-around (e.g., 337.5 -> 22.5): select angles >= dir_min OR <= dir_max
            mask = (wind_direction >= dir_min) | (wind_direction <= dir_max)
        base_wind[mask] *= factor
    
    # Add seasonal variation
    day_of_year = dates.dayofyear.values
    seasonal_factor = 1 + seasonal_amplitude * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    wind_speed = base_wind * seasonal_factor
    
    # Add storm events
    n_storms = int(n_storms_per_year * n_years)
    if storm_directions is None:
        storm_directions = [prevailing_direction - 20, prevailing_direction, prevailing_direction + 20]
    
    storm_indices = np.random.choice(n_samples - storm_duration_range[1], 
                                    min(n_storms, n_samples - storm_duration_range[1]), 
                                    replace=False)
    storm_dirs = np.random.choice(storm_directions, len(storm_indices))
    storm_spreads = np.random.normal(0, 15, len(storm_indices))
    
    for idx, storm_dir, spread in zip(storm_indices, storm_dirs, storm_spreads):
        duration = np.random.randint(storm_duration_range[0], storm_duration_range[1] + 1)
        peak_intensity = np.random.uniform(storm_intensity_range[0], storm_intensity_range[1])
        
        for step in range(duration):
            if idx + step < n_samples:
                # Update direction
                wind_direction[idx + step] = (storm_dir + spread) % 360
                
                # Update speed with bell-shaped profile
                relative_pos = step / (duration - 1) if duration > 1 else 0.5
                intensity_factor = np.exp(-4 * (relative_pos - 0.5)**2)
                wind_speed[idx + step] += peak_intensity * intensity_factor
    
    # Create DataFrame
    wind_data = pd.DataFrame({
        'datetime': dates,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction
    })
    
    if not create_plots:
        return wind_data

    import matplotlib.pyplot as plt  # lazy import
    # Create diagnostic plots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Time series (first year)
    ax1 = fig.add_subplot(gs[0, :])
    mask_year1 = wind_data['datetime'] < (pd.Timestamp(start_date) + pd.DateOffset(years=1))
    ax1.plot(wind_data.loc[mask_year1, 'datetime'], 
             wind_data.loc[mask_year1, 'wind_speed'],
             linewidth=0.5, alpha=0.7, color='steelblue')
    ax1.axhline(np.percentile(wind_speed, 95), color='orange', linestyle='--',
                linewidth=1.5, label='95th percentile', alpha=0.7)
    ax1.set_xlabel('Date (Year 1)', fontsize=11)
    ax1.set_ylabel('Wind Speed (m/s)', fontsize=11)
    ax1.set_title('Wind Speed Time Series (First Year)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Wind Rose
    ax2 = fig.add_subplot(gs[1, 0], projection='polar')
    direction_bins = np.arange(0, 361, 22.5)
    direction_counts, _ = np.histogram(wind_direction, bins=direction_bins)
    direction_freq = direction_counts / len(wind_direction) * 100
    theta = np.radians(direction_bins[:-1] + 11.25)
    width = np.radians(22.5)
    bars = ax2.bar(theta, direction_freq, width=width, bottom=0, alpha=0.7,
                   color='steelblue', edgecolor='black', linewidth=0.5)
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_title('Wind Direction Frequency\n(Wind Rose)', fontsize=11, 
                  fontweight='bold', pad=20)
    
    # Plot 3: Directional speed distribution (8 sectors)
    ax3 = fig.add_subplot(gs[1, 1])
    wind_data['direction_sector'] = pd.cut(
        wind_data['wind_direction'],
        bins=[-22.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5],
        labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
        include_lowest=True
    )
    wind_data.loc[wind_data['wind_direction'] > 337.5, 'direction_sector'] = 'N'
    
    sector_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    wind_data_plot = wind_data[wind_data['direction_sector'].notna()]
    bp = ax3.boxplot(
        [wind_data_plot[wind_data_plot['direction_sector'] == sector]['wind_speed'].values 
         for sector in sector_order],
        labels=sector_order,
        patch_artist=True,
        showfliers=False
    )
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax3.set_xlabel('Direction Sector', fontsize=11)
    ax3.set_ylabel('Wind Speed (m/s)', fontsize=11)
    ax3.set_title('Wind Speed by Direction', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Speed histogram by sector
    ax4 = fig.add_subplot(gs[1, 2])
    west_mask = (wind_direction >= 225) & (wind_direction <= 315)
    other_mask = ~west_mask
    ax4.hist(wind_speed[west_mask], bins=50, alpha=0.6, color='coral',
             label=f'SW-W sector', density=True, edgecolor='black', linewidth=0.5)
    ax4.hist(wind_speed[other_mask], bins=50, alpha=0.6, color='steelblue',
             label='Other sectors', density=True, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax4.set_ylabel('Probability Density', fontsize=11)
    ax4.set_title('Wind Speed Distribution\nby Sector', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'Synthetic Directional Wind Data ({n_years:.1f} Years)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    return wind_data, {'fig': fig, 'axes': (ax1, ax2, ax3, ax4)}
