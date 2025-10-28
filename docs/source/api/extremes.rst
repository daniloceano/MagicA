Extreme Values Analysis
=======================

The ExtremesAnalyzer class provides comprehensive tools for extreme value analysis, including return period and return value calculations for time series data.

Overview
--------

ExtremesAnalyzer enables:

- **Return value analysis**: Calculate values expected to be exceeded once every T time units
- **Return period analysis**: Determine the average time interval between exceedances
- **Block maxima extraction**: Extract annual/monthly/seasonal maxima for GEV analysis
- **Peaks over threshold (POT)**: Extract exceedances for GPD analysis
- **Flexible time handling**: Support for pandas datetime, numeric arrays, and uniform spacing
- **Visualization**: Return level plots with empirical and theoretical curves

Common Use Cases
----------------

- **Climate extremes**: 100-year flood, maximum temperature
- **Wind engineering**: Design wind speeds with specific return periods
- **Ocean engineering**: Extreme wave heights and storm surges
- **Hydrology**: Design rainfall and flood levels
- **Risk assessment**: Probability of extreme events

Class Reference
---------------

.. autoclass:: magica.core.ExtremesAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Quick Start
-----------

Basic Usage with Pandas Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    import magica as ma
    
    # Create time series with datetime index
    dates = pd.date_range('1950-01-01', '2023-12-31', freq='D')
    wind_speeds = np.random.weibull(2.5, len(dates)) * 15 + 5
    series = pd.Series(wind_speeds, index=dates)
    
    # Load data and create extremes analyzer
    processor = ma.read_data(series)
    extremes = processor.get_extremes_analyzer(time_unit='years')
    
    # Fit GEV distribution (common for block maxima)
    extremes.fit_distribution('genextreme')
    
    # Calculate return values
    rv_50 = extremes.return_value(50)   # 50-year return value
    rv_100 = extremes.return_value(100) # 100-year return value
    
    print(f"50-year return value: {rv_50:.2f} m/s")
    print(f"100-year return value: {rv_100:.2f} m/s")

Using Separate Time and Value Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    import magica as ma
    
    # Separate arrays for times and values
    times = pd.date_range('2000-01-01', periods=1000, freq='D')
    values = np.random.weibull(2, 1000) * 10
    
    # Load data and provide times separately
    processor = ma.read_data(values)
    extremes = processor.get_extremes_analyzer(times=times, time_unit='years')
    
    # Fit distribution and analyze
    extremes.fit_distribution('gumbel_r')
    
    # Calculate return period for specific value
    rp = extremes.return_period(25.0)
    print(f"A value of 25 m/s has a return period of {rp:.1f} years")

Core Methods
------------

Distribution Fitting
~~~~~~~~~~~~~~~~~~~~

Fit extreme value distributions to your data:

.. code-block:: python

    # Common distributions for extremes
    extremes.fit_distribution('genextreme')  # Generalized Extreme Value (GEV)
    extremes.fit_distribution('gumbel_r')    # Gumbel (Type I extreme)
    extremes.fit_distribution('weibull_min') # Weibull minimum
    extremes.fit_distribution('weibull_max') # Weibull maximum

**Recommended distributions:**

- **GEV (genextreme)**: Most flexible, encompasses Gumbel, Fréchet, and Weibull
- **Gumbel (gumbel_r)**: Common for environmental extremes
- **Weibull**: Good for wind speeds and structural loads

Return Value Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate the value expected to be exceeded once every T time units:

.. code-block:: python

    # Single return value
    rv_100 = extremes.return_value(100)
    print(f"100-year return value: {rv_100:.2f}")
    
    # Multiple return values
    periods = [10, 20, 50, 100, 500]
    return_values = extremes.return_value(periods)
    
    for period, value in zip(periods, return_values):
        print(f"{period}-year: {value:.2f}")

Return Period Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Determine the average time between exceedances of a given value:

.. code-block:: python

    # Return period for single value
    rp = extremes.return_period(30.0)
    print(f"Value of 30.0 has a return period of {rp:.1f} years")
    
    # Return periods for multiple values
    values = [25, 30, 35, 40]
    periods = extremes.return_period(values)
    
    for val, period in zip(values, periods):
        print(f"Value {val}: {period:.1f}-year return period")

Block Maxima Extraction
~~~~~~~~~~~~~~~~~~~~~~~~

Extract block maxima (or minima) for GEV analysis:

.. code-block:: python

    # Extract annual maxima
    annual_max, times = extremes.extract_block_maxima(block_size='A')
    print(f"Extracted {len(annual_max)} annual maxima")
    
    # Extract monthly maxima
    monthly_max, times = extremes.extract_block_maxima(block_size='M')
    
    # Extract quarterly minima (for minimum extremes)
    quarterly_min, times = extremes.extract_block_maxima(
        block_size='Q',
        method='min'
    )
    
    # Create new analyzer with block maxima
    processor_annual = ma.read_data(pd.Series(annual_max, index=times))
    extremes_annual = processor_annual.get_extremes_analyzer()
    extremes_annual.fit_distribution('genextreme')

**Block sizes** (pandas offset aliases):

- ``'A'`` or ``'Y'``: Annual
- ``'Q'``: Quarterly  
- ``'M'``: Monthly
- ``'W'``: Weekly
- ``'D'``: Daily

Peaks Over Threshold (POT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract exceedances above a threshold for GPD analysis:

.. code-block:: python

    # Extract all peaks over threshold
    peaks, times = extremes.peaks_over_threshold(threshold=20.0)
    print(f"Found {len(peaks)} exceedances")
    
    # Extract peaks with minimum separation (decluster)
    peaks, times = extremes.peaks_over_threshold(
        threshold=20.0,
        min_separation='1D'  # At least 1 day apart
    )
    print(f"Found {len(peaks)} independent peaks")
    
    # Fit GPD to excesses (peaks - threshold)
    excesses = peaks - 20.0
    processor_pot = ma.read_data(excesses)
    processor_pot.fit_distribution('genpareto')

Visualization
-------------

Return Level Plot
~~~~~~~~~~~~~~~~~

Create return level plots to visualize theoretical and empirical return values:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Create return level plot
    fig, ax = extremes.plot_return_levels(
        return_periods=np.logspace(0, 3, 100),  # 1 to 1000 years
        empirical=True
    )
    
    plt.savefig('return_levels.png', dpi=150)
    plt.show()

Custom Visualization
~~~~~~~~~~~~~~~~~~~~

Create custom plots for extreme value analysis:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    # Calculate return values for range of periods
    periods = np.logspace(0, 3, 100)
    rv = extremes.return_value(periods)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(periods, rv, 'b-', linewidth=2)
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('Return Value')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Return Value vs Return Period')
    
    # Add design values
    ax.axhline(extremes.return_value(50), color='r', 
               linestyle='--', label='50-year design value')
    ax.axhline(extremes.return_value(100), color='g',
               linestyle='--', label='100-year design value')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

Advanced Examples
-----------------

Complete Workflow: Annual Maxima Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    import magica as ma
    
    # Load daily wind speed data
    dates = pd.date_range('1980-01-01', '2023-12-31', freq='D')
    wind_speeds = np.random.weibull(2.5, len(dates)) * 12 + 3
    series = pd.Series(wind_speeds, index=dates)
    
    # Create processor and extremes analyzer
    processor = ma.read_data(series)
    extremes = processor.get_extremes_analyzer(time_unit='years')
    
    # Extract annual maxima
    annual_max, max_times = extremes.extract_block_maxima(block_size='A')
    
    # Fit GEV to annual maxima
    processor_annual = ma.read_data(pd.Series(annual_max, index=max_times))
    extremes_annual = processor_annual.get_extremes_analyzer()
    extremes_annual.fit_distribution('genextreme')
    
    # Calculate design values
    design_periods = [10, 20, 50, 100, 500]
    design_values = extremes_annual.return_value(design_periods)
    
    print("Design Wind Speeds (Annual Maxima - GEV):")
    print("=" * 50)
    for period, value in zip(design_periods, design_values):
        print(f"{period:>3}-year: {value:>6.2f} m/s")
    
    # Plot return levels
    fig, ax = extremes_annual.plot_return_levels()
    plt.savefig('annual_maxima_return_levels.png', dpi=150)
    plt.show()

POT Analysis with GPD
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    import magica as ma
    from scipy import stats
    
    # Load data
    dates = pd.date_range('2000-01-01', '2023-12-31', freq='H')
    wave_heights = np.random.weibull(2, len(dates)) * 3 + 0.5
    series = pd.Series(wave_heights, index=dates)
    
    # Create extremes analyzer
    processor = ma.read_data(series)
    extremes = processor.get_extremes_analyzer(time_unit='years')
    
    # Define threshold (e.g., 90th percentile)
    threshold = np.percentile(wave_heights, 90)
    print(f"Threshold (90th percentile): {threshold:.2f} m")
    
    # Extract peaks over threshold with declustering
    peaks, peak_times = extremes.peaks_over_threshold(
        threshold=threshold,
        min_separation='12H'  # At least 12 hours between peaks
    )
    
    print(f"Found {len(peaks)} independent peaks")
    
    # Calculate excesses
    excesses = peaks - threshold
    
    # Fit GPD to excesses
    processor_gpd = ma.read_data(excesses)
    processor_gpd.fit_distribution('genpareto')
    
    # Get GPD parameters
    params = processor_gpd.get_fitted_params()
    print(f"GPD parameters: shape={params[0]:.3f}, loc={params[1]:.3f}, scale={params[2]:.3f}")

Summary Statistics
~~~~~~~~~~~~~~~~~~

Get comprehensive statistics for your extreme value analysis:

.. code-block:: python

    # Get summary
    summary = extremes.get_summary_statistics()
    
    print("Extreme Value Analysis Summary:")
    print("=" * 50)
    print(f"Data points: {summary['data_length']}")
    print(f"Time span: {summary['time_span']:.1f} {summary['time_unit']}")
    print(f"Max value: {summary['max_value']:.2f}")
    print(f"Min value: {summary['min_value']:.2f}")
    print(f"Mean value: {summary['mean_value']:.2f}")
    print(f"Distribution: {summary['distribution']}")
    
    if 'start_date' in summary:
        print(f"Period: {summary['start_date']} to {summary['end_date']}")

Time Handling
-------------

Pandas Series with Datetime Index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most convenient format - ExtremesAnalyzer automatically detects datetime information:

.. code-block:: python

    import pandas as pd
    import magica as ma
    
    # Create series with datetime index
    dates = pd.date_range('2000-01-01', periods=1000, freq='D')
    values = [...]  # your data
    series = pd.Series(values, index=dates)
    
    # Datetime info automatically detected
    processor = ma.read_data(series)
    extremes = processor.get_extremes_analyzer(time_unit='years')

Separate Time Array
~~~~~~~~~~~~~~~~~~~~

Provide times separately if your data isn't in pandas Series format:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import magica as ma
    
    # Separate arrays
    times = pd.date_range('2000-01-01', periods=1000, freq='H')
    values = np.array([...])  # your data
    
    # Provide times to extremes analyzer
    processor = ma.read_data(values)
    extremes = processor.get_extremes_analyzer(times=times, time_unit='years')

Numeric Time Array
~~~~~~~~~~~~~~~~~~

Use numeric values (e.g., decimal years) when datetime isn't needed:

.. code-block:: python

    import numpy as np
    import magica as ma
    
    # Numeric times (e.g., years)
    times = np.linspace(1980, 2023, 1000)
    values = np.array([...])  # your data
    
    processor = ma.read_data(values)
    extremes = processor.get_extremes_analyzer(times=times, time_unit='years')

No Time Information
~~~~~~~~~~~~~~~~~~~

If no time information is provided, uniform spacing is assumed:

.. code-block:: python

    import numpy as np
    import magica as ma
    
    # Just values
    values = np.array([...])
    
    processor = ma.read_data(values)
    extremes = processor.get_extremes_analyzer()
    # Return periods will be in units of observation count

Time Units
~~~~~~~~~~

Specify the time unit for return period calculations:

.. code-block:: python

    # Different time units
    extremes_years = processor.get_extremes_analyzer(time_unit='years')
    extremes_days = processor.get_extremes_analyzer(time_unit='days')
    extremes_hours = processor.get_extremes_analyzer(time_unit='hours')
    extremes_months = processor.get_extremes_analyzer(time_unit='months')

Intelligent POT Threshold Selection
------------------------------------

MagicA provides an automated method for finding optimal POT thresholds that balance the need for:
- High enough threshold to capture only extreme values
- Sufficient sample size for reliable fitting (typically ≥50 independent samples)
- Statistical independence through appropriate declustering

The ``find_optimal_pot_threshold()`` method systematically searches for optimal parameters.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    import magica as ma
    from magica.utils import generate_wind_data
    
    # Generate sample data
    wind_series = generate_wind_data(n_years=30, freq='D', random_seed=42)
    
    # Create extremes analyzer
    processor = ma.read_data(wind_series)
    extremes = processor.get_extremes_analyzer(time_unit='years')
    
    # Find optimal threshold automatically
    result = extremes.find_optimal_pot_threshold(
        min_samples=50,           # Target minimum independent samples
        percentile_min=90,        # Start searching at 90th percentile
        percentile_max=99,        # Stop at 99th percentile
        min_separation_hours=48,  # Minimum time between independent events
        max_separation_hours=120, # Maximum separation to try
        verbose=True              # Show search progress
    )
    
    # Use the results
    if result['success']:
        print(f"Optimal threshold: {result['threshold']:.2f}")
        print(f"Percentile: {result['percentile']:.1f}%")
        print(f"Separation: {result['separation_hours']} hours")
        print(f"Independent exceedances: {result['n_independent']}")
        
        # Access the exceedances for further analysis
        exceedances = result['exceedances']
        exceedance_times = result['exceedance_times']

Search Strategies
~~~~~~~~~~~~~~~~~

The method supports two search strategies controlled by the ``vary_first`` parameter:

**Strategy 1: Vary Percentile First (default)**

Tests each separation time with decreasing percentiles (99→90):

.. code-block:: python

    result = extremes.find_optimal_pot_threshold(
        min_samples=50,
        vary_first='percentile',  # Default strategy
        min_separation_hours=48,
        max_separation_hours=120,
        separation_step_hours=24
    )
    
    # Search order:
    # 1. Try separation=48h: test p99, p98, ..., p90
    # 2. Try separation=72h: test p99, p98, ..., p90
    # 3. Try separation=96h: test p99, p98, ..., p90
    # 4. Try separation=120h: test p99, p98, ..., p90

**Strategy 2: Vary Separation First**

Tests each percentile with increasing separations (48h→120h):

.. code-block:: python

    result = extremes.find_optimal_pot_threshold(
        min_samples=50,
        vary_first='separation',  # Alternative strategy
        percentile_min=90,
        percentile_max=99
    )
    
    # Search order:
    # 1. Try p99: test 48h, 72h, 96h, 120h
    # 2. Try p98: test 48h, 72h, 96h, 120h
    # 3. Continue until min_samples achieved

Customization Examples
~~~~~~~~~~~~~~~~~~~~~~

**Conservative Threshold (Higher Percentiles)**

For critical applications requiring very extreme values:

.. code-block:: python

    result = extremes.find_optimal_pot_threshold(
        min_samples=30,          # Lower requirement acceptable
        percentile_min=95,       # Only test high percentiles
        percentile_max=99.5,
        percentile_step=0.5,     # Finer search steps
        min_separation_hours=48,
        max_separation_hours=72,
        verbose=True
    )

**Sub-Daily Events (Shorter Separations)**

For phenomena with shorter temporal dependence:

.. code-block:: python

    result = extremes.find_optimal_pot_threshold(
        min_samples=80,          # More samples needed
        percentile_min=85,       # Lower percentiles
        percentile_max=95,
        min_separation_hours=6,  # 6-hour minimum separation
        max_separation_hours=24,
        separation_step_hours=6
    )

**Synoptic-Scale Events (Longer Separations)**

For meteorological events with multi-day persistence:

.. code-block:: python

    result = extremes.find_optimal_pot_threshold(
        min_samples=50,
        percentile_min=90,
        percentile_max=99,
        min_separation_hours=72,   # 3-day minimum
        max_separation_hours=168,  # Up to 7 days
        separation_step_hours=24
    )

Return Value Structure
~~~~~~~~~~~~~~~~~~~~~~

The method returns a dictionary with comprehensive information:

.. code-block:: python

    result = extremes.find_optimal_pot_threshold(min_samples=50)
    
    # Always present:
    result['success']              # bool: True if min_samples achieved
    result['threshold']            # float: Optimal threshold value
    result['percentile']           # float: Percentile of threshold
    result['separation_hours']     # float: Time separation used
    result['n_raw_exceedances']    # int: Exceedances before declustering
    result['n_independent']        # int: Independent exceedances after declustering
    result['exceedances']          # array: Independent exceedance values
    result['exceedance_times']     # array: Times of independent exceedances
    result['iterations']           # int: Number of iterations performed
    
    # Optional (if min_samples not achieved):
    result['warning']              # str: Warning message

Best-Effort Fallback
~~~~~~~~~~~~~~~~~~~~

If the minimum sample size cannot be achieved, the method returns the best result found:

.. code-block:: python

    result = extremes.find_optimal_pot_threshold(
        min_samples=100,  # May be too high
        percentile_min=90,
        percentile_max=99
    )
    
    if not result['success']:
        print(f"Warning: {result['warning']}")
        print(f"Achieved {result['n_independent']} samples (target: 100)")
        print(f"Best threshold: {result['threshold']:.2f}")
        
        # Decision: use best-effort or adjust requirements
        if result['n_independent'] >= 80:  # 80% of target
            print("Proceeding with reduced sample size")
        else:
            print("Consider lowering min_samples or using longer data period")

Integration with POT Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete workflow using automated threshold selection:

.. code-block:: python

    import magica as ma
    from magica.utils import generate_wind_data
    
    # Generate data
    wind_series = generate_wind_data(n_years=30, freq='D')
    
    # Create analyzer
    processor = ma.read_data(wind_series)
    extremes = processor.get_extremes_analyzer(time_unit='years')
    
    # Find optimal threshold
    result = extremes.find_optimal_pot_threshold(min_samples=50)
    
    if result['success']:
        # Extract exceedances
        exceedances = result['exceedances']
        threshold = result['threshold']
        
        # Calculate excesses for GPD fitting
        excesses = exceedances - threshold
        
        # Fit GPD distribution
        gpd_processor = ma.read_data(excesses)
        gpd_processor.fit_distribution('genpareto')
        
        # Get parameters
        params = gpd_processor.get_fitted_params()
        shape, loc, scale = params
        
        print(f"GPD parameters: shape={shape:.3f}, scale={scale:.3f}")
        
        # Calculate return values
        # (requires POT-specific return value calculations)

Directional Analysis Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply to each directional sector separately:

.. code-block:: python

    from magica.utils import generate_directional_wind_data
    import magica as ma
    
    # Generate directional data
    wind_data = generate_directional_wind_data(n_years=10, freq='H')
    
    # Define sectors
    sectors = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    sector_results = {}
    
    for sector in sectors:
        # Filter by sector
        sector_mask = wind_data['sector_name'] == sector
        sector_series = wind_data.loc[sector_mask, 'wind_speed']
        
        # Analyze
        processor = ma.read_data(sector_series)
        extremes = processor.get_extremes_analyzer(time_unit='years')
        
        # Find optimal threshold for this sector
        result = extremes.find_optimal_pot_threshold(
            min_samples=50,
            percentile_min=90,
            percentile_max=99,
            min_separation_hours=48,
            max_separation_hours=120,
            verbose=False
        )
        
        sector_results[sector] = result
        
        status = "✓" if result['success'] else "⚠"
        print(f"{status} {sector}: threshold={result['threshold']:.2f} m/s, "
              f"n={result['n_independent']}")

Directional Extreme Value Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``plot_directional_return_values()`` method creates polar plots showing how return values vary by direction, which is essential for wind engineering and offshore structure design.

**Key Features:**

- ✅ Flexible layout: separate subplots or single overlay plot
- ✅ Customizable colors and return periods
- ✅ Automatic subplot arrangement based on number of periods
- ✅ Returns figure and axes for further customization
- ✅ Handles missing/failed sectors gracefully

**Example 1: Separate Subplots (Default)**

.. code-block:: python

    import magica as ma
    import pandas as pd
    
    # After fitting distributions for each directional sector
    # (See "Directional Analysis" section for complete workflow)
    
    # Prepare directional results dictionary
    directional_results = {}
    
    for sector in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
        # sector_data contains wind speeds for this direction
        # center_deg is the center angle of the sector (0° for N, 45° for NE, etc.)
        
        processor = ma.read_data(sector_data)
        extremes = processor.get_extremes_analyzer(time_unit='years')
        extremes.fit_distribution('gumbel_r')
        
        # Calculate return values
        return_values = {
            10: extremes.return_value(10),
            20: extremes.return_value(20),
            50: extremes.return_value(50),
            100: extremes.return_value(100)
        }
        
        directional_results[sector] = {
            'center_deg': center_deg,
            'return_values': return_values,
            'success': True
        }
    
    # Create polar plots (4 subplots in 2x2 grid)
    fig, axes = extremes.plot_directional_return_values(
        directional_results=directional_results,
        return_periods=[10, 20, 50, 100]
    )
    plt.show()

**Example 2: Overlay Mode**

All return periods on a single polar plot with legend:

.. code-block:: python

    # Single plot with all periods overlaid
    fig, ax = extremes.plot_directional_return_values(
        directional_results=directional_results,
        return_periods=[10, 50, 100],
        overlay=True,
        title='Design Wind Speeds by Direction'
    )
    plt.show()

**Example 3: Custom Colors and Fewer Periods**

.. code-block:: python

    # Two periods with custom colors (creates single row)
    fig, axes = extremes.plot_directional_return_values(
        directional_results=directional_results,
        return_periods=[20, 100],
        colors=['#FF6B6B', '#4ECDC4'],  # Custom color scheme
        figsize=(14, 6)
    )
    plt.show()

**Example 4: Single Period**

.. code-block:: python

    # Single period creates one subplot
    fig, ax = extremes.plot_directional_return_values(
        directional_results=directional_results,
        return_periods=[100],
        colors=['#C73E1D']
    )
    
    # Further customize the returned figure
    ax.set_title('100-Year Design Wind Speed', fontsize=16)
    plt.tight_layout()
    plt.savefig('design_wind_100yr.png', dpi=300)

**Layout Rules:**

- **1 period**: Single subplot
- **2 periods**: Single row (1×2)
- **3-4 periods**: Grid layout (2×2)
- **Overlay mode**: Always single plot regardless of number of periods

**Input Format:**

The ``directional_results`` dictionary must have this structure:

.. code-block:: python

    directional_results = {
        'sector_name': {
            'center_deg': float,        # Center angle in degrees (0-360)
            'return_values': {          # Dict mapping return period to value
                10: 25.3,
                20: 28.1,
                50: 31.5,
                100: 34.2
            },
            'success': bool             # Whether fit was successful
        },
        # ... more sectors ...
    }

**Use Cases:**

- **Wind engineering**: Design wind loads by direction for structures
- **Offshore design**: Directional wave height return values
- **Wind turbine siting**: Extreme wind assessment
- **Code compliance**: IEC 61400-1 and other standards

Best Practices
--------------

Distribution Selection
~~~~~~~~~~~~~~~~~~~~~~

1. **GEV (genextreme)**: Use for block maxima (annual, monthly, etc.)
2. **Gumbel (gumbel_r)**: Good approximation when GEV shape parameter is near zero
3. **Weibull**: Often appropriate for wind speeds and structural loads
4. **GPD (genpareto)**: Use for peaks over threshold (POT) analysis

Block Maxima vs POT
~~~~~~~~~~~~~~~~~~~~

**Block Maxima (GEV):**

- ✅ Simple and robust
- ✅ Well-established theory
- ✅ Good for long records
- ⚠️ Wastes data (only uses maxima)
- ⚠️ Requires long record (20+ years recommended)

**Peaks Over Threshold (GPD):**

- ✅ Uses more data
- ✅ Better for short records
- ✅ Can focus on truly extreme values
- ⚠️ Threshold selection can be subjective
- ⚠️ Requires declustering

Sample Size Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Annual maxima**: Minimum 20-30 years for reliable estimates
- **Monthly maxima**: Minimum 5-10 years
- **POT**: Aim for 50-100 independent exceedances

Data Quality
~~~~~~~~~~~~

1. **Check for stationarity**: Extreme value theory assumes stationary data
2. **Remove trends**: Detrend data if significant trends exist
3. **Quality control**: Remove outliers and errors
4. **Independence**: Ensure observations are independent (or decluster POT)

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider parameter uncertainty in your return value estimates:

.. code-block:: python

    # Bootstrap for confidence intervals
    from scipy import stats
    
    # Fit distribution
    extremes.fit_distribution('genextreme')
    params = extremes.fitted_params
    
    # Bootstrap resampling
    n_boot = 1000
    rv_boot = []
    
    for _ in range(n_boot):
        # Resample with replacement
        boot_sample = np.random.choice(extremes.data, size=len(extremes.data), replace=True)
        
        # Fit and calculate return value
        boot_processor = ma.read_data(boot_sample)
        boot_processor.fit_distribution('genextreme')
        boot_extremes = ExtremesAnalyzer(boot_processor)
        boot_extremes.fitted_params = boot_processor.get_fitted_params()
        boot_extremes.distribution_name = 'genextreme'
        rv_boot.append(boot_extremes.return_value(100))
    
    # Calculate confidence intervals
    ci_lower = np.percentile(rv_boot, 2.5)
    ci_upper = np.percentile(rv_boot, 97.5)
    
    print(f"100-year return value: {extremes.return_value(100):.2f}")
    print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

See Also
--------

- :doc:`core` - MagicAdjuster and DataProcessor documentation
- :doc:`auto_fitter` - Automatic distribution selection
- :doc:`/tutorials/index` - Tutorials and examples
- :doc:`/quickstart` - Quick start guide

References
----------

- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
- Beirlant, J., et al. (2004). *Statistics of Extremes: Theory and Applications*. Wiley.
- Gumbel, E.J. (1958). *Statistics of Extremes*. Columbia University Press.
