Monte Carlo Stability Analysis
==============================

The Monte Carlo stability analysis is a powerful method to determine the minimum sample size needed for reliable parameter estimation and to assess how statistical tests behave with different sample sizes.

Overview
--------

The ``monte_carlo_fit()`` method performs repeated subsampling and distribution fitting to:

1. Identify the minimum sample size where parameters stabilize
2. Track how goodness-of-fit tests evolve with sample size
3. Detect the "large sample size effect" where p-values become unreliable
4. Provide empirical distributions for robust inference

Method Signature
----------------

.. code-block:: python

    def monte_carlo_fit(
        self,
        sizes: Optional[List[int]] = None,
        n_repeats: int = 20,
        tests: List[str] = ['ks'],
        stability_method: str = 'aggregate',
        fig_output_path: Optional[str] = None,
        plot_type: str = 'series',
        sampling: str = 'random',
        seed: Optional[int] = None,
        min_size: int = 50,
        max_size: Optional[int] = None,
        n_sizes: int = 10,
        distribution_params: Optional[Tuple] = None,
        **kwargs
    ) -> xr.Dataset

Parameters
----------

Sample Size Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

**sizes** : List[int], optional
    Explicit list of sample sizes to test. If not provided, sizes are automatically generated based on ``min_size``, ``max_size``, and ``n_sizes``.
    
    Example: ``[100, 200, 500, 1000, 2000]``

**min_size** : int, default=50
    Minimum sample size to test (used only if ``sizes`` is not provided).

**max_size** : int, optional
    Maximum sample size to test. Defaults to the size of the original dataset.

**n_sizes** : int, default=10
    Number of different sample sizes to test (used only if ``sizes`` is not provided).

Monte Carlo Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

**n_repeats** : int, default=20
    Number of independent subsamples to draw for each sample size. Higher values provide better statistical estimates but increase computation time.
    
    - For quick exploration: 10-20 repeats
    - For production analysis: 30-50 repeats
    - For publication-quality results: 50-100 repeats

**sampling** : str, default='random'
    Sampling strategy for creating subsamples. Options:
    
    - ``'random'``: Independent random draws without replacement (most common)
    - ``'bootstrap'``: Random draws with replacement (useful for uncertainty quantification)
    - ``'disjoint'``: Non-overlapping partitions (best for temporal/spatial data)

**seed** : int, optional
    Random seed for reproducibility. Always set this for reproducible analyses.

Goodness-of-Fit Tests
~~~~~~~~~~~~~~~~~~~~~

**tests** : List[str], default=['ks']
    List of goodness-of-fit tests to perform. Options:
    
    - ``'ks'``: Kolmogorov-Smirnov test (p-value based)
    - ``'chi2'``: Chi-square test (p-value based)
    - ``'rmse'``: Root Mean Square Error (distance metric)
    
    **⭐ Recommendation**: Always include ``'rmse'`` for stability detection, as it shows clearer convergence behavior than p-value based tests.
    
    Example: ``tests=['ks', 'chi2', 'rmse']``

Stability Detection
~~~~~~~~~~~~~~~~~~~

**stability_method** : str, default='aggregate'
    Method for detecting stability points. Options:
    
    - ``'aggregate'``: Median-based detection with tolerance windows (default, robust)
    - ``'detect'``: Coefficient of variation (CV) based detection
    - ``None``: No stability detection

    See `Stability Detection Methods`_ for detailed explanations.

Visualization
~~~~~~~~~~~~~

**fig_output_path** : str, optional
    Path where the summary figure should be saved. If provided, automatically generates a 2×3 grid showing parameter and test evolution.
    
    Example: ``'monte_carlo_results.png'``

**plot_type** : str, default='series'
    Style of plots to generate:
    
    - ``'series'``: Line plots with median and IQR shading
    - ``'boxplots'``: Box plots showing full distribution per size

Advanced Parameters
~~~~~~~~~~~~~~~~~~~

**distribution_params** : Tuple, optional
    If provided, uses these fixed parameters instead of fitting. Useful for testing how well the method recovers known parameters.

**kwargs**
    Additional keyword arguments passed to the goodness-of-fit tests (e.g., ``bins='doane'`` for chi-square test).

Return Value
------------

Returns an ``xarray.Dataset`` containing:

**Dimensions**

- ``sizes``: Sample sizes tested (length = ``n_sizes``)
- ``repeats``: Repetition index (length = ``n_repeats``)

**Data Variables**

- ``param_0, param_1, ...``: Fitted distribution parameters for each (size, repeat) combination
- ``ks_statistic, ks_pvalue``: Kolmogorov-Smirnov test results (if ``'ks'`` in tests)
- ``chi2_statistic, chi2_pvalue``: Chi-square test results (if ``'chi2'`` in tests)
- ``rmse``: Root mean square error values (if ``'rmse'`` in tests)

**Attributes**

- ``distribution``: Name of the fitted distribution
- ``original_data_size``: Size of the original dataset
- ``sampling_method``: Sampling strategy used
- ``bins_method``: Binning method used for chi-square test
- ``stability_points``: Dictionary with detected stability information
- ``figure_path``: Path to saved figure (if ``fig_output_path`` was provided)
- ``created_by``: Identifier for the method

Stability Detection Methods
----------------------------

Two methods are available for detecting when parameters and tests stabilize:

Aggregate Method (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**How it works:**

1. For each parameter/test, computes the **median** across repeats for each sample size
2. Uses a **sliding window** (default: 4 consecutive sizes) to check for stability
3. Checks if the **range** (max - min) within the window is below a tolerance threshold
4. For parameters: tolerance = 0.1% of the overall range
5. For tests: tolerance = 0.1 (absolute)

**Advantages:**

- Robust to outliers (uses median)
- Clear interpretation (looks for flat regions)
- Works well with any number of repeats

**Result format:**

.. code-block:: python

    stability_points = {
        'rmse': {'size': 600, 'index': 3, 'cv_at_stability': None},
        'param_0': {'size': 800, 'index': 4, 'cv_at_stability': None}
    }

**When to use:** Default choice for most analyses.

Detect Method
~~~~~~~~~~~~~

**How it works:**

1. For each parameter/test, computes the **coefficient of variation** (CV = std/mean) across repeats
2. Checks if CV stays below a threshold (default: 0.1) for a consecutive window of sizes
3. Window size is typically 25% of the number of sizes tested
4. Saves the CV value at the stability point

**Advantages:**

- Provides quantitative measure of variability (CV)
- Good for assessing precision of estimates
- Sensitive to early stabilization

**Result format:**

.. code-block:: python

    stability_points = {
        'rmse': {'size': 600, 'index': 3, 'cv_at_stability': 0.087},
        'param_0': {'size': 800, 'index': 4, 'cv_at_stability': 0.095}
    }

**When to use:** When you need to report the coefficient of variation at stability.

Interpreting Stability Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each stability point contains:

- ``size``: The sample size where stability is first detected (``None`` if not detected)
- ``index``: The index in the ``sizes`` list (``None`` if not detected)
- ``cv_at_stability``: Coefficient of variation at stability (only for ``'detect'`` method)

If ``size`` is ``None``, stability was not detected in the tested range. This can happen when:

- The tested sample sizes are too small
- The parameter/test has high variability
- More repeats are needed (increase ``n_repeats``)

Recommended Practices
---------------------

Choosing Sample Sizes
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # For small to medium datasets (< 5000 samples)
    sizes = [50, 100, 200, 400, 600, 800]
    
    # For large datasets (> 10000 samples) - CPS method
    sizes = [100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000]

Using RMSE for Stability Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**⭐ Always include RMSE in your tests for stability detection:**

.. code-block:: python

    results = adjuster.monte_carlo_fit(
        sizes=[100, 200, 500, 1000, 2000],
        n_repeats=50,
        tests=['ks', 'chi2', 'rmse'],  # RMSE is crucial!
        sampling='random',
        seed=42
    )

**Why RMSE is recommended:**

1. **Monotonic decrease**: RMSE decreases smoothly as sample size increases
2. **Clear convergence**: Stabilization point is visually obvious
3. **Robust**: Less affected by sampling variability than p-values
4. **Direct measure**: Measures actual fit quality, not hypothesis test power
5. **No inflation**: Unlike p-values, doesn't become artificially small with large samples

P-values (KS, Chi-square) can be erratic and are subject to the "large sample size effect" where they become very small even for good fits when the sample is large.

Accessing Stability Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get stability points
    stability = results.attrs['stability_points']
    
    # Recommended size (based on RMSE)
    rmse_stability = stability['rmse']
    optimal_size = rmse_stability['size']
    
    if optimal_size is not None:
        print(f"RMSE stabilizes at n = {optimal_size}")
        print(f"This is the recommended minimum sample size")
    else:
        print("No clear stability detected - try larger sample sizes")

Complete Example
----------------

.. code-block:: python

    import numpy as np
    import magica as ma
    
    # Load data
    data = np.random.weibull(2, 10000) * 8 + 2
    processor = ma.read_data(data)
    processor.fit_distribution('weibull_min')
    
    # Run comprehensive Monte Carlo analysis
    results = processor.monte_carlo_fit(
        sizes=[100, 200, 400, 600, 800, 1000, 1500, 2000],
        n_repeats=50,
        tests=['ks', 'chi2', 'rmse'],  # Include RMSE!
        stability_method='aggregate',
        sampling='random',
        seed=42,
        fig_output_path='monte_carlo_analysis.png'
    )
    
    # Check stability
    stability = results.attrs['stability_points']
    
    # RMSE-based recommendation (most reliable)
    rmse_size = stability['rmse']['size']
    print(f"⭐ RMSE stabilizes at n = {rmse_size} (RECOMMENDED)")
    
    # Other metrics
    for metric in ['ks', 'chi2', 'param_0']:
        size = stability[metric]['size']
        if size:
            print(f"   {metric}: stabilizes at n = {size}")
        else:
            print(f"   {metric}: no clear stability detected")
    
    # Use the recommended size for robust inference
    optimal_size = rmse_size if rmse_size else 1000
    print(f"\nRecommended subsample size: {optimal_size}")

Sampling Strategies
-------------------

Random Sampling (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each subsample is drawn **independently without replacement** from the original data.

**When to use:**

- General purpose analysis
- Independent observations
- Standard stability analysis

**Example:**

.. code-block:: python

    results = processor.monte_carlo_fit(
        sampling='random',
        seed=42
    )

Bootstrap Sampling
~~~~~~~~~~~~~~~~~~

Each subsample is drawn **with replacement**, allowing the same observation to appear multiple times.

**When to use:**

- Uncertainty quantification
- Confidence interval estimation
- When subsample size > original data size

**Example:**

.. code-block:: python

    results = processor.monte_carlo_fit(
        sampling='bootstrap',
        seed=42
    )

Disjoint Sampling
~~~~~~~~~~~~~~~~~

Data is divided into **non-overlapping partitions**. Each observation is used exactly once per sample size.

**When to use:**

- Time series data
- Spatially correlated data
- When you want lower variance between repeats
- More representative of data structure

**Example:**

.. code-block:: python

    results = processor.monte_carlo_fit(
        sampling='disjoint',
        seed=42
    )

Large Sample Size Effect
-------------------------

For very large datasets (>10,000 samples), statistical tests become extremely powerful, causing:

- P-values become very small even for negligible deviations
- Difficulty distinguishing statistical from practical significance
- Over-rejection of perfectly adequate distributions

**Solution**: Use the Monte Carlo CPS (Coefficient/P-value/Sample size) method to find the optimal subsample size where:

1. Parameters have converged (stable estimates)
2. P-values haven't yet inflated (interpretable tests)
3. RMSE has stabilized (good fit quality)

See the :doc:`/tutorials/magic_adjuster_tutorial` for a complete demonstration.

See Also
--------

- :doc:`/tutorials/magic_adjuster_tutorial`: Complete tutorial with real data examples
- :doc:`core`: API reference for MagicAdjuster class
- :doc:`/tutorials/auto_fitter_tutorial`: Automatic distribution selection
