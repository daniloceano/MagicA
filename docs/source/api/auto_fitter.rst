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
   **Best Practice**: Always use **RMSE** as the primary selection criterion. RMSE is robust across all sample sizes and avoids the "large sample size effect" that makes p-values unreliable with large datasets (>10,000 samples).

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
    
    # Create AutoFitter with RMSE criterion (recommended)
    auto_fitter = processor.get_auto_fitter(criterion='rmse')
    
    # Find best distribution
    best_result = auto_fitter.fit_best_distribution()
    
    print(f"Best distribution: {best_result['distribution']}")
    print(f"RMSE: {best_result['rmse']:.6f}")
    print(f"Parameters: {best_result['parameters']}")

Testing All Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get all available distributions
    all_dists = AutoFitter.get_all_available_distributions()
    print(f"Total distributions available: {len(all_dists)}")
    
    # Test all distributions (takes longer)
    auto_fitter = processor.get_auto_fitter(
        candidates=all_dists,
        criterion='rmse'
    )
    
    best = auto_fitter.fit_best_distribution()
    print(f"Best from all 113+: {best['distribution']}")

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
    
    # For synthetic data, optionally filter by p-value
    good_fits = {d: r for d, r in successful.items() 
                 if r['ks_pvalue'] > 0.05}
    
    # Display top 5
    for i, (dist, result) in enumerate(list(good_fits.items())[:5], 1):
        print(f"{i}. {dist}: RMSE={result['rmse']:.6f}")

Using the Best Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you've found the best distribution, you can use it like a regular MagicAdjuster:

.. code-block:: python

    # Get the adjuster for best distribution
    best_adjuster = auto_fitter.get_best_adjuster()
    
    # Calculate statistics
    mean = best_adjuster.stats(moments='m')
    p95 = best_adjuster.ppf(0.95)  # 95th percentile
    
    # Perform goodness-of-fit tests
    ks_result = best_adjuster.goodness_of_fit('ks')
    rmse_result = best_adjuster.goodness_of_fit('rmse')
    
    # Monte Carlo stability analysis
    mc_results = best_adjuster.monte_carlo_fit(
        tests=['ks', 'chi2', 'rmse'],
        n_repeats=100,
        fig_output_path='stability.png'
    )

Selection Criteria
------------------

Available Criteria
~~~~~~~~~~~~~~~~~~

The ``criterion`` parameter accepts:

- **'rmse'** (recommended): Root Mean Square Error - robust for all sample sizes
- **'aic'**: Akaike Information Criterion - balances fit and complexity
- **'bic'**: Bayesian Information Criterion - penalizes complexity more than AIC
- **'ks_pvalue'**: Kolmogorov-Smirnov p-value - statistical significance
- **'chi2_pvalue'**: Chi-square p-value - histogram-based test

When to Use Each
~~~~~~~~~~~~~~~~

**Use RMSE when:**

- ✅ Working with real-world data
- ✅ Sample size is large (>10,000)
- ✅ You want consistent, reliable results
- ✅ Practical fit quality matters

**Use p-values when:**

- ⚠️ Working with synthetic/controlled data
- ⚠️ Sample size is moderate (<1,000)
- ⚠️ Statistical significance is required
- ⚠️ Educational/demonstration purposes

**Use AIC/BIC when:**

- 📊 Comparing model complexity
- 📊 Theoretical model selection
- 📊 Need to balance fit vs. parameters

.. warning::
   **Large Sample Size Effect**: With large datasets (>10,000 observations), goodness-of-fit tests (KS, Chi-square) tend to reject even excellent fits. Their p-values become unreliable. **Always use RMSE for large datasets.**
   
   See the MagicAdjuster tutorial section on "Large Sample Size Effect" for detailed explanation.

Result Dictionary
-----------------

Each distribution's results contain:

.. code-block:: python

    {
        'distribution': str,      # Distribution name
        'success': bool,          # Whether fitting succeeded
        'parameters': tuple,      # Fitted parameters
        'rmse': float,           # Root mean square error
        'aic': float,            # Akaike Information Criterion
        'bic': float,            # Bayesian Information Criterion
        'ks_statistic': float,   # KS test statistic
        'ks_pvalue': float,      # KS test p-value
        'chi2_statistic': float, # Chi-square statistic
        'chi2_pvalue': float,    # Chi-square p-value
        'adjuster': MagicAdjuster  # Fitted adjuster instance
    }

Default Distributions
---------------------

The default candidate list includes 16 stable, commonly-used distributions:

.. code-block:: python

    default_distributions = [
        'norm',           # Normal
        'lognorm',        # Log-normal
        'expon',          # Exponential
        'weibull_min',    # Weibull (minimum)
        'gamma',          # Gamma
        'beta',           # Beta
        'chi2',           # Chi-square
        'rayleigh',       # Rayleigh
        'uniform',        # Uniform
        'logistic',       # Logistic
        'gumbel_r',       # Gumbel (right)
        'exponweib',      # Exponentiated Weibull
        'genextreme',     # Generalized Extreme Value
        'pareto',         # Pareto
        'maxwell',        # Maxwell
        'rice'            # Rice
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
    
    print(f"Best distribution: {best['distribution']}")
    print(f"RMSE: {best['rmse']:.6f}")
    print(f"AIC: {best['aic']:.2f}")
    print(f"KS p-value: {best['ks_pvalue']:.6f}")

Comparing Multiple Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Test with different criteria
    criteria = ['rmse', 'aic', 'bic']
    
    for criterion in criteria:
        fitter = processor.get_auto_fitter(criterion=criterion)
        best = fitter.fit_best_distribution()
        print(f"{criterion.upper()}: {best['distribution']}")

Filtering by P-value (Synthetic Data Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # For synthetic data, you can filter by p-value
    auto_fitter = processor.get_auto_fitter(criterion='rmse')
    auto_fitter.fit_all_distributions()
    
    comparison = auto_fitter.get_comparison_table(sort_by='rmse')
    
    # Get successful fits
    successful = [(d, r) for d, r in comparison.items() if r['success']]
    
    # Filter by p-value > 0.05 (good statistical fit)
    good_fits = [(d, r) for d, r in successful if r['ks_pvalue'] > 0.05]
    
    print(f"Distributions with p > 0.05: {len(good_fits)}")
    print(f"Top 3 by RMSE (p > 0.05):")
    for i, (dist, result) in enumerate(good_fits[:3], 1):
        print(f"  {i}. {dist}: RMSE={result['rmse']:.6f}, p={result['ks_pvalue']:.4f}")

Best Practices
--------------

1. **Always use RMSE** as the primary criterion for real-world data
2. **Start with default distributions** (faster), then try comprehensive if needed
3. **Create custom lists** for domain-specific applications (e.g., wind, rainfall)
4. **Filter by p-value only for synthetic data** with moderate sample sizes
5. **Check multiple criteria** to verify consistency in distribution selection
6. **Use the best adjuster** for further analysis (Monte Carlo, goodness-of-fit)

See Also
--------

- :doc:`core` - MagicAdjuster and DataProcessor documentation
- :doc:`monte_carlo` - Monte Carlo stability analysis
- :doc:`/tutorials/auto_fitter_tutorial` - Complete AutoFitter tutorial
- :doc:`/tutorials/magic_adjuster_tutorial` - MagicAdjuster tutorial with large sample size effect
