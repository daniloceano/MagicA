Quick Start
===========

MagicA (Magic Adjustment) is a Python package for statistical data adjustment with a focus on distribution fitting and Monte Carlo stability analysis.

Installation
------------

Install MagicA using pip:

.. code-block:: bash

    pip install magica

Basic Usage
-----------

Distribution Fitting
~~~~~~~~~~~~~~~~~~~~

Start by fitting a distribution to your data:

.. code-block:: python

    import numpy as np
    import magica as ma
    
    # Generate sample data (Weibull distribution)
    data = np.random.weibull(2, 1000)
    
    # Create processor and fit distribution
    processor = ma.read_data(data)
    processor.fit_distribution('weibull_min')
    
    # Get fitted parameters
    params = processor.get_fitted_params()
    print(f"Fitted parameters: {params}")

Goodness-of-Fit Testing
~~~~~~~~~~~~~~~~~~~~~~~

Evaluate how well your distribution fits the data:

.. code-block:: python

    # Perform Kolmogorov-Smirnov test
    ks_result = processor.goodness_of_fit('ks')
    print(f"KS p-value: {ks_result['p_value']:.4f}")
    
    # Perform chi-square test
    chi2_result = processor.goodness_of_fit('chi2')
    print(f"Chi-square p-value: {chi2_result['p_value']:.4f}")
    
    # Calculate RMSE (recommended)
    rmse_result = processor.goodness_of_fit('rmse')
    print(f"RMSE: {rmse_result['rmse']:.6f}")

.. tip::
   **RMSE is the most reliable metric** for assessing fit quality, especially with large datasets where p-values can be misleading due to the "large sample size effect".

Automatic Distribution Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use AutoFitter to automatically find the best distribution:

.. code-block:: python

    # Create AutoFitter with RMSE criterion (recommended)
    auto_fitter = processor.get_auto_fitter(criterion='rmse')
    
    # Find best distribution
    best = auto_fitter.fit_best_distribution()
    
    print(f"Best distribution: {best['distribution']}")
    print(f"RMSE: {best['rmse']:.6f}")
    print(f"AIC: {best['aic']:.2f}")

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
    extremes.fit_distribution('genextreme')
    
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

Monte Carlo Stability Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determine the minimum sample size needed for stable parameter estimation:

.. code-block:: python

    # Fit a distribution first
    processor.fit_distribution('weibull_min')
    
    # Run Monte Carlo analysis with automatic figure generation
    results = processor.monte_carlo_fit(
        n_repeats=100,
        tests=['ks', 'chi2', 'rmse'],  # Always include RMSE!
        fig_output_path='stability_analysis.png'
    )
    
    # Access results with xarray
    ks_pvalues = results['ks_pvalue']
    rmse_values = results['rmse']
    param_medians = results['param_0'].median(dim='repeats')
    
    # Check stability points
    stability = results.attrs['stability_points']
    
    # RMSE is the most reliable stability indicator
    if 'rmse' in stability:
        print(f"Recommended minimum size (RMSE): {stability['rmse']['size']}")

.. important::
   **Always include 'rmse' in your tests** - RMSE provides the most reliable stability detection with smooth, monotonic convergence, unlike p-values which can be erratic.

Working with xarray Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Monte Carlo analysis returns an xarray Dataset for easy data manipulation:

.. code-block:: python

    # Select data for specific sample size
    size_200_results = results.sel(sizes=200)
    
    # Calculate statistics across repeats
    param_std = results['param_0'].std(dim='repeats')
    rmse_median = results['rmse'].median(dim='repeats')
    
    # Plot results directly
    import matplotlib.pyplot as plt
    results['rmse'].plot(x='sizes')
    plt.title('RMSE Convergence')
    plt.show()
    
    # Convert to pandas for further analysis
    df = results.to_dataframe()

Advanced Examples
-----------------

Custom Binning Strategy
~~~~~~~~~~~~~~~~~~~~~~~

Control the binning strategy for chi-square tests:

.. code-block:: python

    # Use Scott's rule for binning
    results = processor.monte_carlo_fit(
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
    results_random = processor.monte_carlo_fit(
        sampling='random',
        n_repeats=100,
        tests=['ks', 'rmse']
    )
    
    # Bootstrap sampling (with replacement)
    results_bootstrap = processor.monte_carlo_fit(
        sampling='bootstrap',
        n_repeats=100,
        tests=['ks', 'rmse']
    )
    
    # Disjoint sampling (no overlap)
    results_disjoint = processor.monte_carlo_fit(
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
    
    results = processor.monte_carlo_fit(
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
    annual_max, times = extremes.extract_block_maxima(block_size='A')
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

.. code-block:: python

    import pandas as pd
    
    # Create hourly time series
    dates = pd.date_range('2010-01-01', '2023-12-31', freq='H')
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
        min_separation='12H'  # Minimum 12 hours between peaks
    )
    
    print(f"Found {len(peaks)} independent peaks above {threshold:.2f} m")
    
    # Fit GPD to excesses
    excesses = peaks - threshold
    processor_pot = ma.read_data(excesses)
    processor_pot.fit_distribution('genpareto')

.. code-block:: python

    # Use known Weibull parameters (shape=2, loc=0, scale=1)
    known_params = (2.0, 0.0, 1.0)
    
    results = processor.monte_carlo_fit(
        distribution_params=known_params,
        n_repeats=150,
        tests=['chi2', 'ks', 'rmse']
    )

Custom Fitting Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply parameter constraints during fitting:

.. code-block:: python

    # Fix location parameter for Weibull distribution
    results = processor.monte_carlo_fit(
        n_repeats=100,
        tests=['ks', 'rmse'],
        fit_kwargs={'floc': 0}  # Force location = 0
    )
    
    # Multiple constraints
    results = processor.monte_carlo_fit(
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
    size_100_data = results.sel(sizes=100)
    
    # Calculate median across repeats
    param_medians = results['param_0'].median(dim='repeats')
    
    # Check when parameters stabilize
    stability = results.attrs['stability_points']
    if 'param_0' in stability:
        print(f"Parameter 0 stabilizes at size: {stability['param_0']['size']}")
    
    # RMSE stability (most reliable)
    if 'rmse' in stability:
        print(f"RMSE stabilizes at size: {stability['rmse']['size']}")

AutoFitter Results
~~~~~~~~~~~~~~~~~~

AutoFitter returns a dictionary with comprehensive fit information:

.. code-block:: python

    best_result = {
        'distribution': 'weibull_min',
        'success': True,
        'parameters': (2.1, 0.0, 5.3),
        'rmse': 0.012345,
        'aic': 1234.5,
        'bic': 1245.6,
        'ks_statistic': 0.023,
        'ks_pvalue': 0.456,
        'chi2_statistic': 12.3,
        'chi2_pvalue': 0.234,
        'adjuster': <MagicAdjuster instance>
    }

Generating Summary Figures
---------------------------

Automatic Summary Plots
~~~~~~~~~~~~~~~~~~~~~~~

By default no figure is created (faster). Provide a `fig_output_path` to save a
2×3 summary figure (first row: up to 3 parameters; second row: test metrics):

.. code-block:: python

    # Generate and save figure with series style panels
    results = processor.monte_carlo_fit(
        tests=['ks', 'chi2', 'rmse'],
        n_repeats=100,
        fig_output_path='stability_summary.png'
    )

    # Boxplot style
    results = processor.monte_carlo_fit(
        tests=['ks', 'rmse'],
        n_repeats=100,
        plot_type='boxplots',
        fig_output_path='stability_boxplots.png'
    )

    # Path stored in attributes
    print(results.attrs.get('figure_path'))  # saved file path

The red dashed vertical line in each panel marks the first sample size where
the corresponding parameter or test metric meets the stability criterion.

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

1. **Use RMSE as primary criterion** for real-world data
2. **Start with AutoFitter's default list** (16 stable distributions) for quick analysis
3. **Test comprehensive set** (113+ distributions) when you need the absolute best fit
4. **Create domain-specific lists** (e.g., wind, rainfall) for faster, targeted analysis
5. **Avoid p-value-only selection** with large datasets (>10,000 samples)

Monte Carlo Analysis
~~~~~~~~~~~~~~~~~~~~~

1. **Always include 'rmse' in tests** - most reliable stability indicator
2. **Use 100+ repeats** for robust stability detection
3. **Choose appropriate sampling**:
   - `'random'` - General purpose, allows overlap
   - `'bootstrap'` - With replacement, good for uncertainty
   - `'disjoint'` - No overlap, limited by data size
4. **Generate summary figures** with `fig_output_path` for visual inspection
5. **Check RMSE stability first**, then validate with other metrics

Large Sample Size Effect
~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   With large datasets (>10,000 observations), goodness-of-fit tests (KS, Chi-square) tend to reject even excellent fits. **Use RMSE for large datasets.**

See the :doc:`tutorials/magic_adjuster_tutorial` section on "Large Sample Size Effect" for detailed explanation and examples.

Extreme Value Analysis
~~~~~~~~~~~~~~~~~~~~~~~

1. **Use GEV (genextreme) for block maxima** - most flexible for annual/monthly maxima
2. **Use GPD (genpareto) for POT** - peaks over threshold analysis
3. **Require sufficient data**: Minimum 20-30 years for annual maxima
4. **Check stationarity**: Extreme value theory assumes stationary data
5. **Decluster POT data**: Use `min_separation` to ensure independence

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
- See :doc:`api/core` for complete API reference
- Check :doc:`contributing` for development guidelines
