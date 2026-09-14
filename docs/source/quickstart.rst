Quick Start
===========

MagicA (Magic Adjustment) is a Python package for statistical data adjustment with a focus on distribution fitting and Monte Carlo stability analysis.

Installation
------------

Install this repository using pip:

.. code-block:: bash

    pip install "git+https://github.com/daniloceano/MagicA.git"

Basic Usage
-----------

.. _qs-fitting:

Distribution Fitting
~~~~~~~~~~~~~~~~~~~~

Load an array-like sample (a list, NumPy array, or pandas Series/DataFrame),
then fit a distribution. ``fit()`` returns an immutable ``FitResult``;
``fit_distribution()`` is an alias with the same return type.

.. code-block:: python

    import numpy as np
    import magica as ma
    
    # Generate sample data (Weibull distribution)
    data = np.random.weibull(2, 1000)
    
    # Create processor and fit distribution
    processor = ma.read_data(data)
    fit = processor.fit('weibull_min')
    
    # Get fitted parameters
    params = fit.params
    print(f"Fitted parameters: {params}")
    print(fit.info)  # Distribution name, parameters, and sample size
    print(processor.get_basic_stats())  # Descriptive statistics

.. _qs-gof:

Goodness-of-Fit Testing
~~~~~~~~~~~~~~~~~~~~~~~

Evaluate how well your distribution fits the data. KS and chi-square return
dictionaries; RMSE, AIC, and BIC return scalar values. MagicA computes RMSE
between the empirical and fitted CDFs. See :doc:`api/core` for the full API.

.. code-block:: python

    # Perform Kolmogorov-Smirnov test
    ks_result = fit.goodness_of_fit('ks')
    print(f"KS p-value: {ks_result['p_value']:.4f}")
    
    # Perform chi-square test
    chi2_result = fit.goodness_of_fit('chi2')
    print(f"Chi-square p-value: {chi2_result['p_value']:.4f}")
    
    # Calculate CDF RMSE
    rmse_result = fit.goodness_of_fit('rmse')
    print(f"RMSE: {rmse_result:.6f}")

.. tip::
   MagicA recommends RMSE as the primary selection criterion. In the empirical
   tests that motivated the package, its curve produced a clearer stability
   point than the p-value curves. Use test results and plots as complementary
   diagnostics, especially in the region of practical interest.

Automatic Distribution Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use AutoFitter to automatically find the best distribution:

.. code-block:: python

    # Use CDF RMSE as the initial ranking criterion
    auto_fitter = processor.get_auto_fitter(criterion='rmse')
    
    # Find best distribution
    best = auto_fitter.fit_best_distribution()
    
    print(f"Best distribution: {best.name}")
    print(f"RMSE: {best.goodness_of_fit('rmse'):.6f}")
    print(f"AIC: {best.goodness_of_fit('aic'):.2f}")

Testing Multiple Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare all available distributions:

.. code-block:: python

    # Get all 113+ available distributions
    all_dists = auto_fitter.get_all_available_distributions()
    
    # Test comprehensive set
    auto_fitter_all = processor.get_auto_fitter(
        candidates=all_dists,
        criterion='rmse'
    )
    best_comprehensive = auto_fitter_all.fit_best_distribution()
    
    # Get comparison table
    comparison = auto_fitter_all.get_comparison_table(sort_by='rmse')
    
    # Show top 5
    for i, (dist, result) in enumerate(list(comparison.items())[:5], 1):
        if result['success']:
            print(f"{i}. {dist}: RMSE={result['rmse']:.6f}")

Extreme Value Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Analyze extreme values and calculate return periods for time series data:

.. code-block:: python

    import pandas as pd
    
    # Create time series with datetime index
    dates = pd.date_range('1980-01-01', '2023-12-31', freq='D')
    wind_speeds = np.random.weibull(2.5, len(dates)) * 15 + 5
    series = pd.Series(wind_speeds, index=dates)
    
    # Load data and create extremes analyzer
    processor = ma.read_data(series)
    extremes = processor.get_extremes_analyzer(time_unit='years')
    
    # Fit GEV distribution (common for extremes)
    annual_max, annual_times = extremes.extract_block_maxima('YE')
    eva_fit = extremes.fit_block_maxima(annual_max, annual_times)
    
    # Calculate return values
    rv_50 = extremes.return_value(50)   # 50-year return value
    rv_100 = extremes.return_value(100) # 100-year return value
    
    print(f"50-year return value: {rv_50:.2f} m/s")
    print(f"100-year return value: {rv_100:.2f} m/s")
    
    # Calculate return period for specific value
    rp = extremes.return_period(30.0)
    print(f"A value of 30 m/s has a return period of {rp:.1f} years")

.. tip::
   **Common extreme value distributions**:
   
   - ``'genextreme'``: Generalized Extreme Value (GEV) - most flexible
   - ``'gumbel_r'``: Gumbel distribution - common for environmental extremes
   - ``'weibull_max'``: Weibull maximum - for maximum extremes

.. _qs-mc:

Monte Carlo Stability Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examine how parameter estimates and diagnostics vary with sample size:

.. code-block:: python

    # Monte Carlo runs on MagicAdjuster, not DataProcessor or FitResult
    from magica.core import MagicAdjuster
    adjuster = MagicAdjuster(processor)
    fit = adjuster.fit_distribution('weibull_min')
    
    # Run Monte Carlo analysis with automatic figure generation
    results = adjuster.monte_carlo_fit(
        n_repeats=100,
        tests=['ks', 'chi2', 'rmse'],  # Compare tests with CDF distance
        fig_output_path='stability_analysis.png'
    )
    
    # Access results with xarray
    ks_pvalues = results['ks_pvalue']
    rmse_values = results['rmse']
    param_medians = results['param_0'].median(dim='repeats')
    
    # Check stability points
    stability = results.attrs['stability_points']
    
    # Inspect the RMSE result for the selected detector
    if 'rmse' in stability:
        print(f"Detected RMSE stability point: {stability['rmse']['size']}")

.. important::
   Include ``'rmse'`` when CDF distance is relevant. Its curve may show a clear
   bend or plateau, but monotonic convergence is not guaranteed. Interpret the
   detected point within the tested grid and compare it with variability across
   repeats and the other diagnostics.

.. _qs-xarray:

Working with xarray Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Monte Carlo analysis returns an xarray Dataset for easy data manipulation:

.. code-block:: python

    # Inspect dimensions and available variables
    print(results.sizes)
    print(list(results.data_vars))

    # Select the first size, the size nearest 200, or a range
    first_size_results = results.isel(sizes=0)
    size_200 = results.sel(sizes=200, method='nearest')
    larger_sizes = results.sel(sizes=slice(200, None))
    
    # Calculate statistics across repeats
    param_mean = results['param_0'].mean(dim='repeats')
    param_std = results['param_0'].std(dim='repeats')
    rmse_median = results['rmse'].median(dim='repeats')
    
    # Plot results directly
    import matplotlib.pyplot as plt
    results['rmse'].plot(x='sizes', hue='repeats', alpha=0.3)
    plt.title('RMSE Convergence')
    plt.show()
    
    # Convert to pandas for further analysis
    df = results.to_dataframe()
    figure_path = results.attrs['figure_path']  # None when no figure was saved

Advanced Examples
-----------------

Custom Binning Strategy
~~~~~~~~~~~~~~~~~~~~~~~

Control the binning strategy for chi-square tests:

.. code-block:: python

    # Use Scott's rule for binning
    results = adjuster.monte_carlo_fit(
        n_repeats=200,
        tests=['chi2', 'rmse'],
        bins='scott',
        fig_output_path='chi2_stability.png'
    )
    
    # Access chi-square test results
    chi2_pvalues = results['chi2_pvalue']

Different Sampling Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose between random, bootstrap, or disjoint sampling:

.. code-block:: python

    # Random sampling (default)
    results_random = adjuster.monte_carlo_fit(
        sampling='random',
        n_repeats=100,
        tests=['ks', 'rmse']
    )
    
    # Bootstrap sampling (with replacement)
    results_bootstrap = adjuster.monte_carlo_fit(
        sampling='bootstrap',
        n_repeats=100,
        tests=['ks', 'rmse']
    )
    
    # Disjoint sampling (no overlap)
    results_disjoint = adjuster.monte_carlo_fit(
        sampling='disjoint',
        n_repeats=20,  # Limited by data size
        tests=['ks', 'rmse']
    )

Pre-calculated Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

Use known distribution parameters instead of fitting:

.. code-block:: python

    # Use known Weibull parameters (shape=2, loc=0, scale=1)
    known_params = (2.0, 0.0, 1.0)
    
    results = adjuster.monte_carlo_fit(
        distribution_params=known_params,
        n_repeats=150,
        tests=['chi2', 'ks', 'rmse']
    )

Block Maxima for Extreme Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract block maxima from time series for GEV analysis:

.. code-block:: python

    import pandas as pd
    
    # Create daily time series
    dates = pd.date_range('1980-01-01', '2023-12-31', freq='D')
    values = np.random.weibull(2.5, len(dates)) * 15 + 5
    series = pd.Series(values, index=dates)
    
    # Create extremes analyzer
    processor = ma.read_data(series)
    extremes = processor.get_extremes_analyzer()
    
    # Extract annual maxima
    annual_max, times = extremes.extract_block_maxima(block_size='YE')
    print(f"Extracted {len(annual_max)} annual maxima")
    
    # Create new analyzer with annual maxima
    processor_annual = ma.read_data(pd.Series(annual_max, index=times))
    extremes_annual = processor_annual.get_extremes_analyzer()
    extremes_annual.fit_distribution('genextreme')
    
    # Calculate design values
    design_values = extremes_annual.return_value([10, 50, 100])
    print(f"10-year: {design_values[0]:.2f}")
    print(f"50-year: {design_values[1]:.2f}")
    print(f"100-year: {design_values[2]:.2f}")
    
    # Plot return levels
    fig, ax = extremes_annual.plot_return_levels()

Peaks Over Threshold Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze extreme values using peaks over threshold (POT):

POT selects observations strictly above the threshold. With ``min_separation``,
MagicA selects the largest value in each time window by default. Set
``peak_selection='first'`` or ``'last'`` to select the first or last observation.
A window starts at the first ungrouped exceedance and excludes its right endpoint;
selected peaks in adjacent windows can be closer than the window duration.
Ties for the maximum keep the first observation. Without exceedances, extraction
returns an empty array and an empty ``DatetimeIndex`` for datetime input.

The same selection option is available in ``find_optimal_pot_threshold``.
See :doc:`api/extremes` for window examples and return types.

.. code-block:: python

    import pandas as pd
    
    # Create hourly time series
    dates = pd.date_range('2010-01-01', '2023-12-31', freq='h')
    wave_heights = np.random.weibull(2, len(dates)) * 3 + 0.5
    series = pd.Series(wave_heights, index=dates)
    
    # Create extremes analyzer
    processor = ma.read_data(series)
    extremes = processor.get_extremes_analyzer(time_unit='years')
    
    # Define threshold (90th percentile)
    threshold = np.percentile(wave_heights, 90)
    
    # Extract peaks with declustering
    peaks, peak_times = extremes.peaks_over_threshold(
        threshold=threshold,
        min_separation='12h',  # Window duration
        peak_selection='max'  # Default; also 'first' or 'last'
    )
    
    print(f"Found {len(peaks)} selected peaks above {threshold:.2f} m")
    
    # Fit GPD to excesses
    excesses = peaks - threshold
    processor_pot = ma.read_data(excesses)
    processor_pot.fit_distribution('genpareto')


.. _qs-constraints:

Custom Fitting Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply parameter constraints during fitting:

.. code-block:: python

    # Fix location parameter for Weibull distribution
    results = adjuster.monte_carlo_fit(
        n_repeats=100,
        tests=['ks', 'rmse'],
        fit_kwargs={'floc': 0}  # Force location = 0
    )
    
    # Multiple constraints
    results = adjuster.monte_carlo_fit(
        n_repeats=100,
        tests=['ks', 'rmse'],
        fit_kwargs={'floc': 0, 'method': 'MLE'}
    )

Understanding Results
---------------------

Monte Carlo Results Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `monte_carlo_fit` method returns an xarray Dataset with:

**Dimensions:**

- **sizes**: Sample sizes tested (e.g., [50, 100, 150, 200, ...])
- **repeats**: Repetition index for each size (e.g., [0, 1, 2, ..., 99])

**Data Variables:**

- **param_0, param_1, ...**: Fitted distribution parameters for each size/repeat
- **ks_statistic, ks_pvalue**: Kolmogorov-Smirnov test results (if requested)
- **chi2_statistic, chi2_pvalue**: Chi-square test results (if requested)
- **rmse**: Root mean square error values (if requested)

**Attributes:**

- **distribution**: Distribution name
- **original_data_size**: Size of original dataset
- **sampling_method**: Sampling strategy used
- **stability_points**: Detected stability points for each variable
- **figure_path**: Path to saved summary figure (if generated)

**Easy Data Access:**

.. code-block:: python

    # Get all RMSE values
    rmse_data = results['rmse']
    
    # Select specific size
    first_size_data = results.isel(sizes=0)
    
    # Calculate median across repeats
    param_medians = results['param_0'].median(dim='repeats')
    
    # Check when parameters stabilize
    stability = results.attrs['stability_points']
    if 'param_0' in stability:
        print(f"Parameter 0 stabilizes at size: {stability['param_0']['size']}")
    
    # RMSE stability result for the selected detector
    if 'rmse' in stability:
        print(f"RMSE stabilizes at size: {stability['rmse']['size']}")

AutoFitter Results
~~~~~~~~~~~~~~~~~~

``fit_best_distribution()`` returns an immutable ``FitResult``:

.. code-block:: python

    best_fit = auto_fitter.fit_best_distribution()
    print(best_fit.name, best_fit.params)
    print(best_fit.goodness_of_fit('rmse'))

``get_comparison_table()`` returns a mapping of distribution names to scalar
metric dictionaries, including ``params`` and ``success``; it stores no adjusters.

Generating Summary Figures
---------------------------

Automatic Summary Plots
~~~~~~~~~~~~~~~~~~~~~~~

By default no figure is created (faster). Provide a `fig_output_path` to save a
2×3 summary figure (first row: up to 3 parameters; second row: test metrics):

.. code-block:: python

    # Generate and save figure with series style panels
    results = adjuster.monte_carlo_fit(
        tests=['ks', 'chi2', 'rmse'],
        n_repeats=100,
        fig_output_path='stability_summary.png'
    )

    # Boxplot style
    results = adjuster.monte_carlo_fit(
        tests=['ks', 'rmse'],
        n_repeats=100,
        plot_type='boxplots',
        fig_output_path='stability_boxplots.png'
    )

    # Path stored in attributes
    print(results.attrs.get('figure_path'))  # saved file path

The red dashed vertical line in each panel marks the recommended size shared
across panels. Individual detection results are in ``stability_points``. If
``primary_metric`` is ``max_size``, the line represents the fallback, not detected stability.

Custom Visualizations
~~~~~~~~~~~~~~~~~~~~~~

You can also create custom visualizations directly with xarray/matplotlib:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Plot RMSE convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each repeat as a thin line
    for i in range(results.sizes['repeats']):
        results['rmse'].isel(repeats=i).plot(ax=ax, alpha=0.3, color='blue')
    
    # Plot median as thick line
    results['rmse'].median(dim='repeats').plot(ax=ax, linewidth=3, color='red', label='Median')
    
    # Mark stability point
    if 'rmse' in results.attrs['stability_points']:
        stable_size = results.attrs['stability_points']['rmse']['size']
        ax.axvline(stable_size, color='green', linestyle='--', linewidth=2, label=f'Stable at {stable_size}')
    
    ax.set_title('RMSE Convergence Analysis')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('custom_rmse_plot.png', dpi=150)
    plt.show()

Synthetic Data Generation
--------------------------

MagicA provides synthetic data generators for creating realistic test datasets. Perfect for tutorials, testing, and method validation.

Generate Wind Speed Data
~~~~~~~~~~~~~~~~~~~~~~~~

Create non-directional wind speed time series with storms and seasonal variations:

.. code-block:: python

    from magica.utils import generate_wind_data
    import matplotlib.pyplot as plt
    
    # Generate 30 years of daily wind data with plots
    wind_series, plots = generate_wind_data(
        n_years=30,
        freq='D',
        mean_wind=8.0,
        weibull_shape=2.5,
        seasonal_amplitude=0.3,
        n_storms_per_year=5,
        storm_duration_range=(2, 5),
        storm_intensity_range=(12, 20),
        random_seed=42,
        create_plots=True
    )
    
    plt.show()
    
    # Use with MagicA
    processor = ma.read_data(wind_series)
    extremes = processor.get_extremes_analyzer(time_unit='years')
    
    print(f"Generated {len(wind_series)} observations")
    print(f"Max wind speed: {wind_series.max():.2f} m/s")

Generate Directional Wind Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create wind data with directional characteristics:

.. code-block:: python

    from magica.utils import generate_directional_wind_data
    
    # Generate 10 years of hourly directional wind data
    wind_data, plots = generate_directional_wind_data(
        n_years=10,
        freq='h',
        mean_wind=8.0,
        prevailing_direction=270,  # West
        prevailing_concentration=1.5,
        directional_speed_factors={
            'W': 1.4,   # Higher speeds from West (fetch effect)
            'SW': 1.4,
            'N': 0.8,   # Lower speeds from North (land effect)
            'NE': 0.8
        },
        storm_directions=[250, 270, 290],  # Storms from W/SW
        random_seed=42,
        create_plots=True
    )
    
    plt.show()
    
    # Access data
    print(wind_data.head())
    print(f"Columns: {wind_data.columns.tolist()}")

**Directional Speed Factors** accept three formats:

- **Sector names**: ``{'W': 1.4, 'SW': 1.4, 'N': 0.8}`` (8 sectors: N, NE, E, SE, S, SW, W, NW)
- **Degree ranges**: ``{(225, 315): 1.4, (0, 90): 0.8}``
- **Numeric centers**: ``{270: 1.4, 0: 0.8}`` (auto-creates ±22.5° range)

Intelligent POT Threshold Selection
------------------------------------

Automatically find optimal POT thresholds that balance extreme values with sample size.

Basic Threshold Search
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from magica.utils import generate_wind_data
    import magica as ma
    
    # Generate sample data
    wind_series = generate_wind_data(n_years=30, freq='D', random_seed=42)
    
    # Create extremes analyzer
    processor = ma.read_data(wind_series)
    extremes = processor.get_extremes_analyzer(time_unit='years')
    
    # Find optimal threshold automatically
    result = extremes.find_optimal_pot_threshold(
        min_samples=50,           # Target number of selected peaks
        percentile_min=90,        # Start at 90th percentile
        percentile_max=99,        # Stop at 99th percentile
        min_separation_hours=48,  # Smallest window duration to try
        max_separation_hours=120,
        verbose=True              # Show search progress
    )
    
    # Use the results
    if result['success']:
        print(f"Optimal threshold: {result['threshold']:.2f} m/s")
        print(f"Selected peaks: {result['n_independent']}")
        
        # Access exceedances for further analysis
        exceedances = result['exceedances']
        exceedance_times = result['exceedance_times']

Custom Search Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

Control the search order with ``vary_first`` parameter:

.. code-block:: python

    # Strategy 1: Vary percentile first (default)
    # Tests each separation with decreasing percentiles
    result1 = extremes.find_optimal_pot_threshold(
        min_samples=50,
        vary_first='percentile'  # Default
    )
    
    # Strategy 2: Vary separation first
    # Tests each percentile with increasing separations
    result2 = extremes.find_optimal_pot_threshold(
        min_samples=50,
        vary_first='separation'
    )

Directional POT Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Apply to each directional sector:

.. code-block:: python

    from magica.utils import generate_directional_wind_data
    
    # Generate directional data
    wind_data = generate_directional_wind_data(n_years=10, freq='h')
    
    # Analyze each sector
    sectors = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    sector_results = {}
    
    for sector in sectors:
        # Filter by sector
        sector_mask = wind_data['sector_name'] == sector
        sector_series = wind_data.loc[sector_mask].set_index('datetime')['wind_speed']
        
        # Find optimal threshold
        processor = ma.read_data(sector_series)
        extremes = processor.get_extremes_analyzer(time_unit='years')
        
        result = extremes.find_optimal_pot_threshold(
            min_samples=50,
            min_separation_hours=48,
            max_separation_hours=120
        )
        
        sector_results[sector] = result
        print(f"{sector}: threshold={result['threshold']:.2f}, "
              f"n={result['n_independent']}")

Directional Return Value Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create polar plots of return values by direction:

.. code-block:: python

    # After fitting distributions for each sector
    directional_results = {}
    
    for sector in sectors:
        # ... (fit distribution and calculate return values) ...
        directional_results[sector] = {
            'center_deg': center_angle,
            'return_values': {10: rv10, 50: rv50, 100: rv100},
            'success': True
        }
    
    # Create polar plots (separate subplots)
    fig, axes = extremes.plot_directional_return_values(
        directional_results=directional_results,
        return_periods=[10, 50, 100]
    )
    
    # Or create overlay plot
    fig, ax = extremes.plot_directional_return_values(
        directional_results=directional_results,
        return_periods=[10, 50, 100],
        overlay=True
    )

Supported Distributions
-----------------------

MagicA supports all continuous distributions from scipy.stats, including:

**Common Distributions:**

- `'norm'` - Normal (Gaussian) distribution
- `'weibull_min'` - Weibull distribution (most common form)
- `'weibull_max'` - Weibull maximum distribution
- `'gamma'` - Gamma distribution
- `'lognorm'` - Log-normal distribution
- `'expon'` - Exponential distribution
- `'rayleigh'` - Rayleigh distribution (wind speed)
- `'beta'` - Beta distribution
- `'chi2'` - Chi-square distribution
- `'uniform'` - Uniform distribution

**Extreme Value Distributions:**

- `'gumbel_r'` - Gumbel (right) distribution
- `'gumbel_l'` - Gumbel (left) distribution  
- `'genextreme'` - Generalized Extreme Value
- `'exponweib'` - Exponentiated Weibull

**Environmental Data Distributions:**

- `'weibull_min'` - Wind speed (most common)
- `'lognorm'` - Rainfall, pollutant concentrations
- `'gamma'` - Rainfall amounts
- `'maxwell'` - Molecular speeds, wave heights
- `'rice'` - Wind with persistent component

**113+ Total Distributions Available** - See ``AutoFitter.get_all_available_distributions()`` for complete list.

Best Practices
--------------

Distribution Selection
~~~~~~~~~~~~~~~~~~~~~~

1. **Use RMSE as MagicA's empirically recommended primary criterion**
2. **Start with AutoFitter's default list** (16 stable distributions) for quick analysis
3. **Expand the candidate set carefully** when the default list is insufficient
4. **Create domain-specific lists** (e.g., wind, rainfall) for faster, targeted analysis
5. **Use p-values as diagnostics**, considering calibration and multiple comparisons

Monte Carlo Analysis
~~~~~~~~~~~~~~~~~~~~~

1. **Use RMSE as the primary stability signal**, following MagicA's empirical recommendation
2. **Increase repeats until the interpretation is insensitive to that choice**
3. **Choose appropriate sampling**:
   - `'random'` - General purpose, allows overlap
   - `'bootstrap'` - With replacement, good for uncertainty
   - `'disjoint'` - No overlap, limited by data size
4. **Generate summary figures** with `fig_output_path` for visual inspection
5. **Compare detector results and inspect the curves visually**

Large Sample Size Effect
~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   As sample size grows, goodness-of-fit tests can detect smaller departures
   from a candidate distribution. There is no universal sample-size cutoff.
   Interpret statistical significance together with discrepancy magnitude,
   location, and practical importance.

See the :doc:`tutorials/magic_adjuster_tutorial` section on "Large Sample Size Effect" for detailed explanation and examples.

Extreme Value Analysis
~~~~~~~~~~~~~~~~~~~~~~~

1. **Use GEV (genextreme) for block maxima** - most flexible for annual/monthly maxima
2. **Use GPD (genpareto) for POT** - peaks over threshold analysis
3. **Require sufficient data**: Minimum 20-30 years for annual maxima
4. **Check stationarity**: Extreme value theory assumes stationary data
5. **Decluster POT data**: Use `min_separation` to group exceedances into windows; assess independence separately

**Time Series Requirements:**

- Use pandas Series with datetime index for automatic time handling
- Or provide times separately via `get_extremes_analyzer(times=...)`
- Specify appropriate `time_unit` ('years', 'days', 'hours', 'months')

Next Steps
----------

- Explore the :doc:`tutorials/index` for detailed examples
- Learn about :doc:`api/auto_fitter` for automatic distribution selection
- Read :doc:`api/monte_carlo` for comprehensive Monte Carlo documentation
- Check :doc:`api/extremes` for complete extreme value analysis documentation
- See :doc:`api/utils` for synthetic data generation and utilities
- Check :doc:`api/core` for complete API reference
- See :doc:`contributing` for development guidelines
