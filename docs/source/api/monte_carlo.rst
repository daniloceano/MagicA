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

**stability_method** : str, default='kneedle'
    Method for detecting when parameters and tests have stabilized. Options:
    
    - ``'kneedle'``: Kneedle algorithm for elbow detection (default, best for RMSE)
    - ``'cv'``: Coefficient of Variation method (robust for erratic metrics)
    - ``'plateau'``: Relative gain heuristic (early "good enough" detection)
    - ``'aggregate'``: Legacy median-based method (still supported)
    - ``None``: No stability detection
    
    **⭐ Default: 'kneedle'** - Recommended for RMSE stability detection as it excels at finding the "point of diminishing returns" in smooth, converging metrics.

    See `Stability Detection Methods`_ for detailed explanations and selection guidance.

**Method-Specific Parameters (via kwargs):**

    For **CV method**:
    
    - ``window_size``: Number of consecutive sizes for validation (default: 25% of n_sizes)
    - ``cv_threshold``: Maximum allowed coefficient of variation (default: 0.1)
    
    For **Kneedle method**:
    
    - ``smooth``: Whether to smooth curves before detection (default: True)
    - ``smoothing_method``: 'savgol' (default) or 'spline'
    
    For **Plateau method**:
    
    - ``consecutive_points``: Number of consecutive points below tolerance (default: 3)
    - ``relative_tolerance``: Maximum relative change threshold (default: 0.01, i.e., 1%)

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
- ``recommended_size``: Sample size where primary metric stabilizes (None if not detected)
- ``primary_metric``: The metric used for recommended_size (typically 'rmse')
- ``stable_pvalue_ks``: KS test p-value at stability point (if 'ks' in tests)
- ``stable_pvalue_chi2``: Chi-square p-value at stability point (if 'chi2' in tests)
- ``stable_rmse``: RMSE value at stability point (if 'rmse' in tests)
- ``n_repeats``: Number of repetitions performed
- ``min_size``: Minimum sample size tested
- ``max_size``: Maximum sample size tested
- ``n_sizes``: Number of sample sizes tested
- ``figure_path``: Path to saved figure (if ``fig_output_path`` was provided)
- ``created_by``: Identifier for the method

Stability Detection Methods
----------------------------

MagicA provides three advanced methods for detecting when parameters and goodness-of-fit metrics have stabilized. Each method has specific strengths depending on the metric behavior.

CV Method (Coefficient of Variation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**How it works:**

1. For each parameter/test, computes the **coefficient of variation** (CV = std/mean) across repeats at each sample size
2. Uses a **sliding window** (default: 25% of n_sizes) to check for stability
3. Checks if CV stays **below a threshold** (default: 0.1) for all sizes in the window
4. Returns the first size where this condition is met

**Mathematical formula:**

.. math::

    CV_n = \frac{\sigma_n}{\mu_n}

Where:
- :math:`\sigma_n` = standard deviation across repeats at size n
- :math:`\mu_n` = mean value across repeats at size n

**Advantages:**

- Robust to erratic metrics (especially p-values)
- Provides quantitative measure of variability
- Good for assessing precision of estimates
- Works well with any number of repeats

**Disadvantages:**

- May be conservative (detects stability later than other methods)
- Less sensitive to smooth convergence patterns

**Parameters:**

- ``window_size``: Number of consecutive sizes for validation (default: max(2, n_sizes // 4))
- ``cv_threshold``: Maximum allowed CV (default: 0.1)

**Result format:**

.. code-block:: python

    stability_points = {
        'rmse': {
            'size': 800,
            'index': 4,
            'cv_at_stability': 0.087,
            'smoothed_curve': None,
            'method': 'cv'
        }
    }

**When to use:** 
- When working with erratic p-values (KS, Chi-square)
- When you need quantitative variability assessment
- When conservative estimates are preferred

Kneedle Method (Elbow Detection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**How it works:**

1. **Normalizes** the curve to [0, 1] range
2. Computes the **reference line** connecting first and last points
3. Calculates **perpendicular distance** from each point to the reference line
4. The **knee/elbow** is at the point with maximum distance
5. Optional: **smooths the curve** before detection using Savitzky-Golay filter or spline

**Mathematical formula:**

.. math::

    D_i = |y_i - (y_0 + (y_n - y_0) \cdot \frac{i}{n})|

Where:
- :math:`D_i` = distance from reference line at point i
- :math:`y_i` = normalized curve value at point i
- :math:`y_0, y_n` = first and last curve values

**Advantages:**

- **Optimal for RMSE**: Excels at finding "point of diminishing returns" in smooth curves
- Clear mathematical definition of "elbow"
- Automatic curve direction detection
- Smoothing reduces noise sensitivity
- Based on published algorithm (Satopää et al., 2011)

**Disadvantages:**

- Requires relatively smooth convergence
- May not work well with highly erratic metrics

**Parameters:**

- ``smooth``: Whether to smooth before detection (default: True, recommended)
- ``smoothing_method``: 'savgol' (default, preserves features) or 'spline'

**Result format:**

.. code-block:: python

    stability_points = {
        'rmse': {
            'size': 600,
            'index': 3,
            'cv_at_stability': None,
            'smoothed_curve': array([...]),  # If smoothing was used
            'method': 'kneedle'
        }
    }

**When to use:**
- **⭐ RECOMMENDED for RMSE stability detection**
- When metrics show smooth, monotonic convergence
- When you want to find the "sweet spot" of diminishing returns
- For production analyses requiring objective elbow detection

**Reference:**

Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). *Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior*. 2011 31st International Conference on Distributed Computing Systems Workshops, 166-171.

Plateau Method (Relative Gain Heuristic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**How it works:**

1. Computes **relative change** between consecutive points:
   
   .. math::
   
       \\Delta_i = \\frac{|y_{i-1} - y_i|}{|y_{i-1}|}

2. Checks if relative change stays **below tolerance** for L consecutive points
3. Returns the first point where this "plateau" condition is met

**Mathematical formula:**

Stability detected when:

.. math::

    \Delta_i < \epsilon \text{ for } L \text{ consecutive points}

Where:
- :math:`\Delta_i` = relative change at point i
- :math:`\epsilon` = relative tolerance (default: 0.01, i.e., 1%)
- :math:`L` = consecutive points threshold (default: 3)

**Advantages:**

- Early detection of "good enough" stabilization
- Intuitive interpretation (% improvement)
- Fast - no smoothing required
- Good for early stopping in iterative analyses

**Disadvantages:**

- May detect premature plateaus
- Sensitive to parameter tuning
- Less robust to noise than other methods

**Parameters:**

- ``consecutive_points``: Number of consecutive points below tolerance (default: 3)
- ``relative_tolerance``: Maximum relative change threshold (default: 0.01)

**Result format:**

.. code-block:: python

    stability_points = {
        'rmse': {
            'size': 500,
            'index': 2,
            'cv_at_stability': None,
            'smoothed_curve': None,
            'method': 'plateau'
        }
    }

**When to use:**
- When computational budget is limited
- For early "good enough" detection
- When slight improvements beyond plateau are acceptable
- In iterative analyses where early stopping is beneficial

Method Selection Guide
~~~~~~~~~~~~~~~~~~~~~~

Choose the appropriate method based on your metric and goals:

+----------------+------------------+---------------------+-------------------------+
| Method         | Best For         | Strengths           | Typical Use Case        |
+================+==================+=====================+=========================+
| **Kneedle**    | RMSE, converging | Objective elbow     | **Production RMSE**     |
|                | parameters       | detection, smooth   | **analysis**            |
|                |                  | curves              | **(RECOMMENDED)**       |
+----------------+------------------+---------------------+-------------------------+
| **CV**         | P-values (KS,    | Robust to noise,    | P-value stability,      |
|                | Chi-square)      | quantifies          | conservative estimates  |
|                |                  | variability         |                         |
+----------------+------------------+---------------------+-------------------------+
| **Plateau**    | Early detection, | Fast, intuitive,    | Quick exploration,      |
|                | limited budget   | early stopping      | iterative tuning        |
+----------------+------------------+---------------------+-------------------------+
| **Aggregate**  | Legacy analyses  | Simple, robust      | Backward compatibility  |
|                |                  |                     |                         |
+----------------+------------------+---------------------+-------------------------+

**Recommended Workflow:**

.. code-block:: python

    # 1. Use Kneedle for RMSE (most reliable)
    results_rmse = processor.monte_carlo_fit(
        tests=['rmse'],
        stability_method='kneedle',
        smooth=True
    )
    
    # 2. Use CV for p-values (robust to noise)
    results_pvalues = processor.monte_carlo_fit(
        tests=['ks', 'chi2'],
        stability_method='cv',
        cv_threshold=0.15  # Can relax for p-values
    )
    
    # 3. Compare stability points
    rmse_stable = results_rmse.attrs['recommended_size']
    print(f"RMSE stable at n = {rmse_stable} (MOST RELIABLE)")

Interpreting Stability Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each stability point contains:

- ``size``: The sample size where stability is first detected (``None`` if not detected)
- ``index``: The index in the ``sizes`` list (``None`` if not detected)
- ``cv_at_stability``: Coefficient of variation at stability (only for ``'cv'`` method)
- ``smoothed_curve``: Smoothed curve data (only for ``'kneedle'`` method with smoothing)
- ``method``: Name of the detection method used

**Accessing stability information:**

.. code-block:: python

    # Get recommended size (based on primary metric, usually RMSE)
    recommended_n = results.attrs['recommended_size']
    primary_metric = results.attrs['primary_metric']
    
    if recommended_n:
        print(f"✓ {primary_metric.upper()} stable at n = {recommended_n}")
        
        # Get quality metrics at stable point
        if 'stable_rmse' in results.attrs:
            print(f"  RMSE at stable point: {results.attrs['stable_rmse']:.4f}")
        if 'stable_pvalue_ks' in results.attrs:
            print(f"  KS p-value at stable point: {results.attrs['stable_pvalue_ks']:.4f}")
    else:
        print("✗ No stability detected - try larger sample sizes")

**If stability is not detected:**

- The tested sample sizes are too small - increase ``max_size``
- The metric has high variability - increase ``n_repeats``
- The threshold is too strict - relax method-specific parameters
- The data distribution may not be appropriate

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

    # Get recommended size (based on primary metric, typically RMSE)
    recommended_n = results.attrs['recommended_size']
    primary_metric = results.attrs['primary_metric']
    
    if recommended_n is not None:
        print(f"✓ {primary_metric.upper()} stabilizes at n = {recommended_n}")
        print(f"  This is the recommended minimum sample size")
        
        # Get quality metrics at the stable point
        print(f"\nMetrics at stable point (n = {recommended_n}):")
        if 'stable_rmse' in results.attrs:
            print(f"  RMSE: {results.attrs['stable_rmse']:.4f}")
        if 'stable_pvalue_ks' in results.attrs:
            print(f"  KS p-value: {results.attrs['stable_pvalue_ks']:.4f}")
        if 'stable_pvalue_chi2' in results.attrs:
            print(f"  Chi-square p-value: {results.attrs['stable_pvalue_chi2']:.4f}")
    else:
        print("✗ No clear stability detected - try larger sample sizes or more repeats")
    
    # Access individual stability points for all metrics
    stability = results.attrs['stability_points']
    for metric, info in stability.items():
        size = info['size']
        method = info['method']
        if size:
            print(f"{metric}: stable at n = {size} (method: {method})")
            if info['cv_at_stability'] is not None:
                print(f"  CV at stability: {info['cv_at_stability']:.3f}")

Complete Example
----------------

.. code-block:: python

    import numpy as np
    import magica as ma
    
    # Load data
    data = np.random.weibull(2, 10000) * 8 + 2
    processor = ma.read_data(data)
    processor.fit_distribution('weibull_min')
    
    # Run comprehensive Monte Carlo analysis with Kneedle method
    results = processor.monte_carlo_fit(
        sizes=[100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000],
        n_repeats=50,
        tests=['ks', 'chi2', 'rmse'],  # Include RMSE!
        stability_method='kneedle',     # Use Kneedle for RMSE
        smooth=True,                    # Smooth curves before detection
        sampling='random',
        seed=42,
        fig_output_path='monte_carlo_analysis.png'
    )
    
    # Check recommended size (based on RMSE with Kneedle)
    recommended_n = results.attrs['recommended_size']
    primary_metric = results.attrs['primary_metric']
    
    print(f"⭐ {primary_metric.upper()} stabilizes at n = {recommended_n} (RECOMMENDED)")
    print(f"   RMSE at stable point: {results.attrs['stable_rmse']:.4f}")
    print(f"   KS p-value at stable point: {results.attrs['stable_pvalue_ks']:.4f}")
    
    # Check all stability points
    stability = results.attrs['stability_points']
    print("\nAll stability points:")
    for metric, info in stability.items():
        size = info['size']
        method = info['method']
        if size:
            print(f"   {metric}: n = {size} (method: {method})")
        else:
            print(f"   {metric}: no clear stability detected")
    
    # Use the recommended size for robust inference
    print(f"\nRecommended subsample size for production: {recommended_n}")

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
