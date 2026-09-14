Monte Carlo Stability Analysis
==============================

Monte Carlo stability analysis examines how fitted parameters and diagnostics
vary across a user-defined grid of sample sizes. It reports points selected by a
stability detector; these points are empirical summaries of the configured run,
not universal minimum sample sizes.

Overview
--------

The ``monte_carlo_fit()`` method performs repeated subsampling and distribution fitting to:

1. Identify where parameter estimates appear stable within the tested grid
2. Track how goodness-of-fit tests evolve with sample size
3. Show how increasing test power affects p-values for small model departures
4. Store the distribution of results across repeated subsamples

Method Signature
----------------

.. code-block:: python

    def monte_carlo_fit(
        self,
        sizes: Optional[List[int]] = None,
        n_repeats: int = 20,
        tests: List[str] = ['ks'],
        stability_method: str = 'kneedle',
        fig_output_path: Optional[str] = None,
        plot_type: str = 'series',
        sampling: str = 'random',
        seed: Optional[int] = None,
        min_size: int = 50,
        max_size: Optional[int] = None,
        n_sizes: int = 10,
        distribution_params: Optional[Tuple] = None,
        **kwargs
    ): ...  # returns xarray.Dataset

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
    Number of subsamples to draw for each sample size. More repeats describe
    Monte Carlo variability more precisely but increase computation time.
    
    - For quick exploration: 10-20 repeats
    - For a more stable summary: 30-50 repeats
    - For a detailed sensitivity analysis: 50-100 repeats

    These ranges are starting points rather than accuracy guarantees. Check
    whether conclusions change when the number of repeats is increased.

**sampling** : str, default='random'
    Sampling strategy for creating subsamples. Options:
    
    - ``'random'``: Random draws without replacement within each draw (most common)
    - ``'bootstrap'``: Random draws with replacement (useful for uncertainty quantification)
    - ``'disjoint'``: Shuffled partitions, non-overlapping within each shuffle

**seed** : int, optional
    Random seed for reproducibility. Always set this for reproducible analyses.

Goodness-of-Fit Tests
~~~~~~~~~~~~~~~~~~~~~

**tests** : List[str], default=['ks']
    List of goodness-of-fit tests to perform. Options:
    
    - ``'ks'``: Kolmogorov-Smirnov test (p-value based)
    - ``'chi2'``: Chi-square test (p-value based)
    - ``'rmse'``: Root Mean Square Error (distance metric)
    
    MagicA recommends ``'rmse'`` as the primary stability signal. In the
    empirical tests that motivated the method, RMSE produced a clearer stability
    point than the p-value curves. KS and chi-square results remain useful as
    complementary diagnostics. AIC and BIC are not calculated by
    ``monte_carlo_fit()`` and their stability was not evaluated in this workflow.
    
    Example: ``tests=['ks', 'chi2', 'rmse']``

Stability Detection
~~~~~~~~~~~~~~~~~~~

**stability_method** : str, default='kneedle'
    Method for detecting when parameters and tests have stabilized. Options:
    
    - ``'kneedle'``: Kneedle algorithm for elbow detection (default)
    - ``'cv'``: Coefficient of Variation method
    - ``'plateau'``: Relative gain heuristic (early "good enough" detection)
    - ``'aggregate'``: Legacy median-based method (still supported)
    - ``None``: No stability detection
    
    Kneedle is useful when the summarized curve has a visible bend. Its result
    can change with the sample-size grid and smoothing configuration.

    See `Stability Detection Methods`_ for detailed explanations and selection guidance.

**Method-Specific Parameters (via kwargs):**

    For **CV method**:
    
    - ``window_size``: Number of consecutive sizes for validation (default: max(2, n_sizes // 4))
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
- ``recommended_size``: Detected size for the primary metric; if none is
  detected, the largest tested size is returned with ``primary_metric='max_size'``
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

MagicA provides three current detection methods plus the legacy aggregate
method. Each applies a different operational definition of stability, so compare
the result with the plotted curves and test its sensitivity to configuration.

CV Method (Coefficient of Variation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**How it works:**

1. For each parameter/test, computes the **coefficient of variation** (CV = std/abs(mean)) across repeats at each sample size
2. Uses a **sliding window** (default: max(2, n_sizes // 4)) to check for stability
3. Checks if CV stays **below a threshold** (default: 0.1) for all sizes in the window
4. Returns the first size where this condition is met

**Mathematical formula:**

.. math::

    CV_n = \frac{\sigma_n}{|\mu_n|}

Where:
- :math:`\sigma_n` = standard deviation across repeats at size n
- :math:`\mu_n` = mean value across repeats at size n

**Advantages:**

- Expresses dispersion relative to the mean
- Provides a quantitative variability threshold
- Can require stability across consecutive tested sizes

**Disadvantages:**

- Becomes unstable or very large when the mean is close to zero
- Can be difficult to interpret for bounded, skewed quantities such as p-values
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

- When relative variability across repeats is meaningful
- When the metric mean stays sufficiently far from zero
- When the CV threshold can be justified and checked for sensitivity

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

- Identifies a point of diminishing change in curves with a visible bend
- Uses an explicit mathematical definition of an elbow
- Automatic curve direction detection
- Optional smoothing can reduce local noise
- Based on published algorithm (Satopää et al., 2011)

**Disadvantages:**

- Requires relatively smooth convergence
- May not work well with highly erratic metrics
- Sensitive to the tested grid and smoothing choices

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

- When metrics show smooth, monotonic convergence
- When you want to identify a geometric bend associated with diminishing changes
- When you can inspect sensitivity to the grid and smoothing settings

**Reference:**

Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). *Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior*. 2011 31st International Conference on Distributed Computing Systems Workshops, 166-171.

Plateau Method (Relative Gain Heuristic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**How it works:**

1. Computes **relative change** between consecutive points:
   
   .. math::
   
       \Delta_i = \frac{|y_{i-1} - y_i|}{|y_{i-1}|}

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
| Method         | Suitable for     | Main signal         | Main limitation         |
+================+==================+=====================+=========================+
| **Kneedle**    | Smooth curves    | Geometric bend      | Grid and smoothing      |
|                | with a bend      |                     | sensitivity             |
+----------------+------------------+---------------------+-------------------------+
| **CV**         | Relative         | Dispersion across   | Mean near zero and      |
|                | variability      | repeats             | bounded metrics         |
+----------------+------------------+---------------------+-------------------------+
| **Plateau**    | Small relative   | Consecutive changes | Tolerance sensitivity   |
|                | changes          | below tolerance     |                         |
+----------------+------------------+---------------------+-------------------------+
| **Aggregate**  | Legacy analyses  | Median-based rule   | Kept for compatibility  |
+----------------+------------------+---------------------+-------------------------+

**Recommended Workflow:**

Use Kneedle on the RMSE curve as the primary analysis, following the empirical
workflow used to develop MagicA. Inspect p-value curves separately because they
answer a different question and commonly respond strongly to increasing sample
size.

.. code-block:: python

    import numpy as np
    import magica as ma

    data = np.random.default_rng(42).weibull(2, 1000)
    processor = ma.read_data(data)
    adjuster = processor._get_adjuster()
    adjuster.fit_distribution('weibull_min')

    # Apply Kneedle to the RMSE curve
    results_rmse = adjuster.monte_carlo_fit(
        tests=['rmse'],
        stability_method='kneedle',
        smooth=True
    )
    
    # Inspect the test outputs as complementary diagnostics
    results_pvalues = adjuster.monte_carlo_fit(
        tests=['ks', 'chi2'],
        stability_method='cv',
        cv_threshold=0.15  # Can relax for p-values
    )
    
    # 3. Compare stability points
    rmse_size = results_rmse.attrs['recommended_size']
    rmse_metric = results_rmse.attrs['primary_metric']
    print(f"RMSE run: {rmse_metric} at n = {rmse_size}")

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

    # Inspect the primary result and distinguish detection from fallback
    primary_size = results.attrs['recommended_size']
    primary_metric = results.attrs['primary_metric']
    
    if primary_metric != 'max_size':
        print(f"{primary_metric.upper()} stability detected at n = {primary_size}")
        
        # Get quality metrics at stable point
        if results.attrs.get('stable_rmse') is not None:
            print(f"  RMSE at stable point: {results.attrs['stable_rmse']:.4f}")
        if results.attrs.get('stable_pvalue_ks') is not None:
            print(f"  KS p-value at stable point: {results.attrs['stable_pvalue_ks']:.4f}")
    else:
        print(f"No stability detected; largest tested size = {primary_size}")

**If stability is not detected:**

- Inspect the curve before changing the configuration
- Test whether increasing ``max_size`` or ``n_repeats`` changes the result
- Examine sensitivity to method-specific thresholds and smoothing
- Reconsider whether the selected metric and candidate distribution suit the goal

Recommended Practices
---------------------

Choosing Sample Sizes
~~~~~~~~~~~~~~~~~~~~~

The following grids are examples. Choose limits and spacing that cover the
sample sizes relevant to the application, then check whether a finer or wider
grid changes the detected point.

.. code-block:: python

    # A compact exploratory grid
    sizes = [50, 100, 200, 400, 600, 800]
    
    # A wider grid for examining later changes
    sizes = [100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000]

Using RMSE for Stability Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use RMSE as the primary stability signal, following MagicA's empirical
recommendation, and track test statistics and p-values alongside it:

.. code-block:: python

    results = adjuster.monte_carlo_fit(
        sizes=[100, 200, 500, 1000, 2000],
        n_repeats=50,
        tests=['ks', 'chi2', 'rmse'],
        sampling='random',
        seed=42
    )

**How to interpret RMSE here:**

1. It directly summarizes CDF distance on the sampled observations.
2. It is not a hypothesis test and therefore does not encode a significance threshold.
3. In the tests that motivated MagicA, its curve produced a clearer stability
   point than the p-value curves.
4. It gives equal weight to the evaluated CDF points and may not emphasize tail discrepancies.
5. It does not penalize model complexity. AIC/BIC can complement distribution
   selection when finite, but they are not implemented as Monte Carlo metrics
   here and their stability has not been evaluated by this workflow.

P-values answer a different question. As sample size grows, tests can detect
smaller departures from the candidate distribution, including departures that
may be negligible for the intended application.

Accessing Stability Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Inspect the primary result and distinguish detection from fallback
    primary_size = results.attrs['recommended_size']
    primary_metric = results.attrs['primary_metric']
    
    if primary_metric != 'max_size':
        print(f"{primary_metric.upper()} stability detected at n = {primary_size}")
        
        # Get quality metrics at the stable point
        print(f"\nMetrics at detected point (n = {primary_size}):")
        if results.attrs.get('stable_rmse') is not None:
            print(f"  RMSE: {results.attrs['stable_rmse']:.4f}")
        if results.attrs.get('stable_pvalue_ks') is not None:
            print(f"  KS p-value: {results.attrs['stable_pvalue_ks']:.4f}")
        if results.attrs.get('stable_pvalue_chi2') is not None:
            print(f"  Chi-square p-value: {results.attrs['stable_pvalue_chi2']:.4f}")
    else:
        print(f"No stability detected; largest tested size = {primary_size}")
    
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
    from magica.core import MagicAdjuster
    adjuster = MagicAdjuster(processor)
    fit = adjuster.fit_distribution('weibull_min')
    
    # Run comprehensive Monte Carlo analysis with Kneedle method
    results = adjuster.monte_carlo_fit(
        sizes=[100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000],
        n_repeats=50,
        tests=['ks', 'chi2', 'rmse'],
        stability_method='kneedle',     # Detect bends in summarized curves
        smooth=True,                    # Smooth curves before detection
        sampling='random',
        seed=42,
        fig_output_path='monte_carlo_analysis.png'
    )
    
    # Inspect whether a primary stability point was detected
    primary_size = results.attrs['recommended_size']
    primary_metric = results.attrs['primary_metric']
    
    if primary_metric == 'max_size':
        print(f"No stability detected; largest tested size = {primary_size}")
    else:
        print(f"{primary_metric.upper()} stability detected at n = {primary_size}")
    
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
    
    # Treat this as a configuration-dependent result, not a universal minimum
    print(f"\nPrimary result: {primary_metric} at n = {primary_size}")

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

    results = adjuster.monte_carlo_fit(
        sampling='random',
        seed=42
    )

Bootstrap Sampling
~~~~~~~~~~~~~~~~~~

Each subsample is drawn **with replacement**, allowing the same observation to appear multiple times.

**When to use:**

- Uncertainty quantification
- Exploring sampling variability when bootstrap assumptions are appropriate
- When subsample size > original data size

**Example:**

.. code-block:: python

    results = adjuster.monte_carlo_fit(
        sampling='bootstrap',
        seed=42
    )

Disjoint Sampling
~~~~~~~~~~~~~~~~~

Indices are shuffled and divided into **non-overlapping partitions within each
shuffle**. Further shuffles can reuse observations when more repeats are needed;
the method does not preserve temporal order or perform temporal segmentation.

**When to use:**

- When non-overlap within each shuffled partition is useful
- When you want to limit reuse before starting another shuffle
- When temporal or spatial ordering is not required

This strategy does not preserve time or spatial structure. Segment or resample
dependent data explicitly when the application requires it.

**Example:**

.. code-block:: python

    results = adjuster.monte_carlo_fit(
        sampling='disjoint',
        seed=42
    )

Large Sample Size Effect
-------------------------

As sample size grows, goodness-of-fit tests gain power and can detect
increasingly small departures from a candidate distribution. There is no
universal sample size at which this begins. Possible consequences include:

- Very small p-values for deviations that may be negligible in practice
- Difficulty distinguishing statistical from practical significance
- Rejection of models that may still be adequate for a specific use

Monte Carlo stability analysis helps describe this behavior by showing how the
following quantities change over a chosen range of sample sizes:

1. Fitted parameter estimates
2. Test statistics and p-values
3. CDF RMSE

A detected point is conditional on the sampled data, grid, number of repeats,
metric, and detector settings. It should be interpreted with domain knowledge
and does not by itself establish an optimal sample size or model adequacy.

See the :doc:`/tutorials/magic_adjuster_tutorial` for a complete synthetic-data demonstration.

See Also
--------

- :doc:`/tutorials/magic_adjuster_tutorial`: Complete tutorial with synthetic data examples
- :doc:`core`: API reference for MagicAdjuster class
- :doc:`/tutorials/auto_fitter_tutorial`: Automatic distribution selection
