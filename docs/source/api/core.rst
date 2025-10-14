Core Module
===========

The core module contains the primary classes for statistical data processing and distribution fitting.

.. automodule:: magica.core
   :members:
   :undoc-members:
   :show-inheritance:

DataProcessor
-------------

The `DataProcessor` class handles data loading, validation, and provides access to distribution fitting capabilities.

.. autoclass:: magica.core.DataProcessor
   :members:
   :undoc-members:
   :show-inheritance:

Automatic Distribution Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DataProcessor provides the `get_auto_fitter()` method to create an AutoFitter instance for automatic distribution selection:

.. code-block:: python

    import numpy as np
    import magica as ma
    
    # Load data
    data = np.random.weibull(2, 1000)
    processor = ma.read_data(data)
    
    # Get AutoFitter for automatic distribution selection
    auto_fitter = processor.get_auto_fitter(criterion='rmse')
    
    # Find best distribution
    best = auto_fitter.fit_best_distribution()
    print(f"Best distribution: {best['distribution']}")

For complete AutoFitter documentation, see :doc:`auto_fitter`.

MagicAdjuster
---------------

The `MagicAdjuster` class is the central component for statistical distribution fitting and Monte Carlo stability analysis.

.. autoclass:: magica.core.MagicAdjuster
   :members:
   :undoc-members:
   :show-inheritance:

Monte Carlo Fitting
~~~~~~~~~~~~~~~~~~~

The `monte_carlo_fit` method performs stability analysis to determine minimum sample sizes for reliable parameter estimation.

.. note::
   For complete documentation including all parameters, stability detection methods, and best practices, see :doc:`monte_carlo`.

**Quick Example:**

.. code-block:: python

    import numpy as np
    from magica.core import MagicAdjuster
    
    # Generate sample data
    data = np.random.weibull(2, 1000)
    
    # Create adjuster and fit distribution
    adjuster = MagicAdjuster(data)
    adjuster.fit_distribution('weibull_min')
    
    # Run Monte Carlo stability analysis
    results = adjuster.monte_carlo_fit(
        sizes=[100, 200, 500, 1000],
        n_repeats=30,
        tests=['ks', 'chi2', 'rmse'],  # Always include RMSE!
        sampling='random',
        seed=42,
        fig_output_path='stability.png'
    )
    
    # Check RMSE stability (most reliable indicator)
    rmse_size = results.attrs['stability_points']['rmse']['size']
    print(f"Recommended minimum sample size: {rmse_size}")

**Return Value:**
**Return Value:**

xarray.Dataset with:

- **Dimensions**: 
  - `sizes`: Sample sizes tested
  - `repeats`: Repetition index for each size

- **Data Variables**: 
  - `param_0, param_1, ...`: Fitted distribution parameters
  - `ks_statistic, ks_pvalue`: Kolmogorov-Smirnov test results (if requested)
  - `chi2_statistic, chi2_pvalue`: Chi-square test results (if requested)
  - `rmse`: Root mean square error values (if requested)

- **Attributes**: 
  - `distribution`: Distribution name
  - `original_data_size`: Size of original dataset
  - `sampling_method`: Sampling strategy used
  - `stability_points`: Detected stability points for each variable (dict with 'size', 'index', 'cv_at_stability')
  - `figure_path`: Path to saved figure (if generated)

.. tip::
   **Always include 'rmse' in your tests** - it provides the most reliable stability detection because it shows smooth, monotonic convergence unlike p-values which can be erratic.

For complete parameter documentation, stability methods, and best practices, see :doc:`monte_carlo`.

Goodness-of-Fit Testing
~~~~~~~~~~~~~~~~~~~~~~~

The class provides comprehensive goodness-of-fit testing with multiple methods:

**Chi-Square Test:**
Tests the hypothesis that data follows the fitted distribution using histogram-based comparison.

**Kolmogorov-Smirnov Test:**
Non-parametric test comparing empirical and theoretical cumulative distribution functions.

**Root Mean Square Error (RMSE):**
Measures the average deviation between empirical and theoretical distributions.

Utility Methods
~~~~~~~~~~~~~~~

**Binning Strategies:**

The class supports multiple binning strategies for histogram-based tests:

- `_calculate_sturges_bins()`: Sturges' rule (log-based)
- `_calculate_rice_bins()`: Rice rule (cube root)
- `_calculate_freedman_diaconis_bins()`: Freedman-Diaconis rule (IQR-based)
- `_calculate_scott_bins()`: Scott's rule (standard deviation-based)
- `_calculate_doane_bins()`: Doane's rule (skewness-adjusted)

**Subsampling:**

- `_generate_subsample_indices()`: Creates index lists for different sampling strategies

Sampling strategies
-------------------

For a didactic, longer discussion of sampling strategies (`random`, `bootstrap`,
and `disjoint`) and practical advice on when to use each, see the Monte Carlo
tutorial: :doc:`/tutorials/monte_carlo`.
