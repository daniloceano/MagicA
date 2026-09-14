AutoFitter - Automatic Distribution Selection
==============================================

The AutoFitter class provides automatic distribution fitting and selection capabilities, testing multiple probability distributions and selecting the best one based on specified criteria.

Overview
--------

AutoFitter automatically:

- Tests multiple probability distributions (default: 16 curated distributions)
- Can test all 113+ SciPy continuous distributions
- Selects best distribution based on RMSE, AIC, BIC, or p-values
- Uses lazy initialization for memory efficiency
- Provides comprehensive comparison tables

.. important::
   MagicA recommends RMSE as the primary criterion. This recommendation is
   empirical: in the tests that motivated the package, RMSE produced a clearer
   stability point than the p-value curves. AIC and BIC are useful complementary
   likelihood criteria when their values are finite, but their stability was not
   evaluated in this workflow.

Class Reference
---------------

.. autoclass:: magica.core.AutoFitter
   :members:
   :undoc-members:
   :show-inheritance:

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import magica as ma
    
    # Load data
    data = np.random.weibull(2, 1000) * 8 + 2
    processor = ma.read_data(data)
    
    # Use CDF RMSE as the initial ranking criterion
    auto_fitter = processor.get_auto_fitter(criterion='rmse')
    
    # Find best distribution
    best_result = auto_fitter.fit_best_distribution()
    
    print(f"Best distribution: {best_result.name}")
    print(f"RMSE: {best_result.goodness_of_fit('rmse'):.6f}")
    print(f"Parameters: {best_result.params}")

Testing All Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from magica.core import AutoFitter

    # Get all available distributions
    all_dists = AutoFitter.get_all_available_distributions()
    print(f"Total distributions available: {len(all_dists)}")
    
    # Test all distributions (takes longer)
    auto_fitter = processor.get_auto_fitter(
        candidates=all_dists,
        criterion='rmse'
    )
    
    best = auto_fitter.fit_best_distribution()
    print(f"Best from all 113+: {best.name}")

Custom Distribution List
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Define domain-specific distributions
    wind_distributions = [
        'weibull_min',    # Most common for wind
        'weibull_max',
        'rayleigh',       # Theoretical wind model
        'lognorm',
        'gamma',
        'rice'
    ]
    
    auto_fitter = processor.get_auto_fitter(
        candidates=wind_distributions,
        criterion='rmse'
    )
    
    best = auto_fitter.fit_best_distribution()

Working with Results
--------------------

Comparison Table
~~~~~~~~~~~~~~~~

Get a comprehensive comparison of all tested distributions:

.. code-block:: python

    # Fit all distributions first
    auto_fitter.fit_all_distributions()
    
    # Get sorted comparison table
    comparison = auto_fitter.get_comparison_table(sort_by='rmse')
    
    # Filter successful fits
    successful = {d: r for d, r in comparison.items() if r['success']}
    
    # Display the first five entries in the RMSE ranking with KS diagnostics
    for i, (dist, result) in enumerate(list(successful.items())[:5], 1):
        print(
            f"{i}. {dist}: RMSE={result['rmse']:.6f}, "
            f"KS p-value={result['ks_pvalue']:.4f}"
        )

Using the Best Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you've found the best distribution, you can use it through the returned ``FitResult``:

.. code-block:: python

    # Get the adjuster for best distribution
    best_fit = auto_fitter.fit_best_distribution()
    
    # Calculate statistics
    mean = best_fit.stats(moments='m')
    p95 = best_fit.ppf(0.95)  # 95th percentile
    
    # Perform goodness-of-fit tests
    ks_result = best_fit.goodness_of_fit('ks')
    rmse_result = best_fit.goodness_of_fit('rmse')
    
    # Monte Carlo requires a fitted MagicAdjuster
    from magica.core import MagicAdjuster
    adjuster = MagicAdjuster(processor)
    adjuster.fit_distribution(best_fit.name)
    mc_results = adjuster.monte_carlo_fit(
        tests=['ks', 'chi2', 'rmse'],
        n_repeats=100,
        fig_output_path='stability.png'
    )

Selection Criteria
------------------

Available Criteria
~~~~~~~~~~~~~~~~~~

The ``criterion`` parameter accepts:

- **'rmse'**: Root Mean Square Error between the fitted and empirical CDFs
- **'aic'**: Akaike Information Criterion - balances fit and complexity
- **'bic'**: Bayesian Information Criterion - penalizes complexity more than AIC
- **'ks_pvalue'**: Kolmogorov-Smirnov p-value
- **'chi2_pvalue'**: Histogram-based chi-square p-value

All continuous distributions exposed by MagicA provide the SciPy ``logpdf``
interface used for AIC and BIC, but fitting or likelihood evaluation can still
fail or return a non-finite value for a particular dataset. Check
``numpy.isfinite`` before comparing these criteria across candidates.

When to Use Each
~~~~~~~~~~~~~~~~

**Use RMSE when:**

- You want the primary criterion recommended by MagicA's empirical workflow
- You want a direct summary of the distance between fitted and empirical CDFs
- You will inspect p-values, plots, and tail behavior as complementary evidence

**Use p-values when:**

- You want a goodness-of-fit diagnostic rather than a model ranking alone
- The test assumptions and the effect of estimating parameters are considered
- You interpret the values together with effect-size and graphical diagnostics

**Use AIC/BIC when:**

- You want likelihood-based comparisons with a penalty for fitted parameters
- Every candidate being compared has a finite likelihood criterion on the same data
- You do not interpret them as Monte Carlo stability measures: the current
  ``monte_carlo_fit()`` workflow does not calculate AIC or BIC

.. warning::
   As sample size grows, goodness-of-fit tests can detect increasingly small
   departures from a candidate distribution. A small p-value may therefore
   coexist with a fit that is adequate for a particular practical purpose. No
   universal sample-size cutoff applies: consider the magnitude and location of
   the discrepancy, especially in the tails, alongside RMSE, AIC/BIC, and plots.

   When parameters are estimated from the same observations used by a test, its
   nominal p-value may also require calibration. Comparing many candidates adds
   a multiple-comparison and selection effect, so do not rank models by p-value
   alone.

Result Dictionary
-----------------

``get_comparison_table()`` and ``fit_all_distributions()`` return scalar
metric dictionaries per candidate (not ``FitResult`` objects). Successful entries contain:

.. code-block:: python

    {
        'distribution': str,      # Distribution name
        'success': bool,          # Whether fitting succeeded
        'params': tuple,      # Fitted parameters
        'rmse': float,           # Root mean square error
        'aic': float,            # Akaike Information Criterion
        'bic': float,            # Bayesian Information Criterion
        'ks_statistic': float,   # KS test statistic
        'ks_pvalue': float,      # KS test p-value
        'chi2_statistic': float, # Chi-square statistic
        'chi2_pvalue': float     # Chi-square p-value
    }

Default Distributions
---------------------

The default candidate list includes 16 stable, commonly-used distributions:

.. code-block:: python

    default_distributions = [
        'weibull_min', 'lognorm', 'gamma', 'norm', 'expon', 'rayleigh',
        'chi2', 'beta', 'uniform', 'logistic', 'gumbel_r', 'pareto',
        'invgamma', 'maxwell', 'triang', 'laplace',
    ]

To test all 113+ available distributions, use:

.. code-block:: python

    all_dists = AutoFitter.get_all_available_distributions()

Examples
--------

Finding Best Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import magica as ma
    
    # Generate wind speed data
    wind_data = np.random.weibull(2.5, 5000) * 10 + 2
    processor = ma.read_data(wind_data)
    
    # Auto-fit with RMSE criterion
    auto_fitter = processor.get_auto_fitter(criterion='rmse')
    best = auto_fitter.fit_best_distribution()
    
    print(f"Best distribution: {best.name}")
    print(f"RMSE: {best.goodness_of_fit('rmse'):.6f}")
    print(f"AIC: {best.goodness_of_fit('aic'):.2f}")
    print(f"KS p-value: {best.goodness_of_fit('ks')['p_value']:.6f}")

Comparing Multiple Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Test with different criteria
    criteria = ['rmse', 'aic', 'bic']
    
    for criterion in criteria:
        fitter = processor.get_auto_fitter(criterion=criterion)
        best = fitter.fit_best_distribution()
        print(f"{criterion.upper()}: {best.name}")

Inspecting P-values
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Inspect p-values alongside the RMSE ranking
    auto_fitter = processor.get_auto_fitter(criterion='rmse')
    auto_fitter.fit_all_distributions()
    
    comparison = auto_fitter.get_comparison_table(sort_by='rmse')
    
    # Get successful fits
    successful = [(d, r) for d, r in comparison.items() if r['success']]
    
    above_threshold = [
        (d, r) for d, r in successful if r['ks_pvalue'] > 0.05
    ]
    
    print(f"Distributions with p > 0.05: {len(above_threshold)}")
    print(f"Top 3 by RMSE (p > 0.05):")
    for i, (dist, result) in enumerate(above_threshold[:3], 1):
        print(f"  {i}. {dist}: RMSE={result['rmse']:.6f}, p={result['ks_pvalue']:.4f}")

The threshold is illustrative; exceeding it does not establish that a model is
correct or make the p-value a model-selection score. Calibration may be needed
when parameters were estimated from the same data.

Best Practices
--------------

1. **Use RMSE as MagicA's empirically recommended primary criterion**
2. **Start with default distributions** (faster), then try comprehensive if needed
3. **Create custom lists** for domain-specific applications (e.g., wind, rainfall)
4. **Treat p-values as diagnostics**, with attention to calibration and multiple comparisons
5. **Check complementary criteria and plots**, including the region or tail that matters
6. **Use FitResult** for distribution evaluation and a fitted MagicAdjuster for Monte Carlo

See Also
--------

- :doc:`core` - MagicAdjuster and DataProcessor documentation
- :doc:`monte_carlo` - Monte Carlo stability analysis
- :doc:`/tutorials/auto_fitter_tutorial` - Complete AutoFitter tutorial
- :doc:`/tutorials/magic_adjuster_tutorial` - MagicAdjuster tutorial with large sample size effect
