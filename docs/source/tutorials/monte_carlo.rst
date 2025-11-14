Monte Carlo Stability Tutorial
==============================

This tutorial explains why Monte Carlo stability analysis is useful and how
the `monte_carlo_fit` routine in MagicA is designed and used.

Why Monte Carlo stability?
--------------------------

Large-sample-size effects occur when very large datasets make statistical tests
extremely powerful: p-values become very small even for effects that are
practically negligible. In other words, with enough data a test can reject the
null hypothesis for differences that have no practical importance. The Monte
Carlo stability workflow therefore looks for a sampling size where the chosen
goodness-of-fit tests start to "pass" in a practical sense and parameter
estimates stop changing much across repeats — that sample size is reported as
the stability point.


Methodology (what we do)
-------------------------

1. Build a grid of sample sizes to test (or use a provided list).
2. For each size, generate multiple subsamples (repeats) from the original data.
3. For each subsample, fit the distribution (unless fixed parameters are supplied).
4. Compute goodness-of-fit metrics (e.g., KS p-value, chi-square p-value, RMSE).
5. Store all results in an xarray Dataset with dimensions `sizes` x `repeats`.
6. Detect stability points using a moving-window criterion on variability (e.g., CV).
7. Optionally save a 2x3 summary figure with red dashed vertical lines marking each
   variable's detected stability sample size.

Inputs and options
------------------

- `sizes`: list or auto-generated grid of sample sizes.
- `n_repeats`: number of subsamples per size.
- `tests`: list of metrics to compute: `['ks','chi2','rmse']`.
- `stability_method`: method for detecting stability points (`'kneedle'` [default], `'cv'`, `'plateau'`).
- `sampling`: sampling strategy (`random`, `bootstrap`, `disjoint`).
- `fit_kwargs`: passed to the underlying fit routine (e.g., `floc=0`).
- `distribution_params`: optional fixed parameters (skip fitting when supplied).
- `fig_output_path`: optional path to save the 2x3 summary figure.
- `plot_type`: `'series'` (median + IQR) or `'boxplots'`.

Stability Detection Methods
----------------------------

MagicA provides three methods for detecting when parameters and goodness-of-fit
metrics have stabilized. Each method has different strengths depending on the
metric behavior.

CV Method (Coefficient of Variation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Monitors the coefficient of variation (CV = std/mean) across
repeats. Stability is detected when CV remains below a threshold for a
consecutive window of sample sizes.

**Mathematical definition**:

.. math::

    CV_n = \\frac{\\sigma_n}{\\mu_n}

Where :math:`\\sigma_n` is the standard deviation and :math:`\\mu_n` is the mean
across repeats at sample size n.

**When to use**:

- Erratic p-values (KS, Chi-square tests)
- When you need quantitative variability assessment
- Conservative estimates preferred

**Parameters** (via kwargs):

- `window_size`: Number of consecutive sizes for validation (default: 25% of n_sizes)
- `cv_threshold`: Maximum allowed CV (default: 0.1)

Kneedle Method (Elbow Detection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Based on the "Kneedle" algorithm (Satopää et al., 2011). Finds the
"elbow" or "knee" point in a curve by:

1. Normalizing the curve to [0, 1]
2. Computing a reference line from first to last point
3. Finding the point with maximum perpendicular distance from the reference line

**Mathematical definition**:

The knee point is where:

.. math::

    \\text{knee} = \\arg\\max_i |y_i - (y_0 + (y_n - y_0) \\cdot \\frac{i}{n})|

Where :math:`y_i` is the normalized curve value at point i.

**When to use**:

- **RMSE and converging parameters** (RECOMMENDED)
- Smooth, monotonic convergence patterns
- Finding "point of diminishing returns"
- Production analyses requiring objective detection

**Parameters** (via kwargs):

- `smooth`: Whether to smooth curves before detection (default: True)
- `smoothing_method`: 'savgol' (preserves features) or 'spline' (default: 'savgol')

**Reference**: Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011).
*Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior*.
2011 31st International Conference on Distributed Computing Systems Workshops, 166-171.

Plateau Method (Relative Gain Heuristic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Detects when relative improvement between consecutive points
becomes negligible. Stability is found when the relative change stays below a
threshold for L consecutive points.

**Mathematical definition**:

Relative change at point i:

.. math::

    \\Delta_i = \\frac{|y_{i-1} - y_i|}{|y_{i-1}|}

Stability when:

.. math::

    \\Delta_i < \\epsilon \\text{ for } L \\text{ consecutive points}

**When to use**:

- Limited computational budget
- Early "good enough" detection
- Iterative analyses with early stopping

**Parameters** (via kwargs):

- `consecutive_points`: Number of points below tolerance (default: 3)
- `relative_tolerance`: Maximum relative change (default: 0.01, i.e., 1%)

Method Selection Guide
~~~~~~~~~~~~~~~~~~~~~~

+----------------+------------------+-------------------------+
| Method         | Best For         | Typical Use Case        |
+================+==================+=========================+
| **Kneedle**    | RMSE,            | Production RMSE         |
|                | parameters       | analysis (RECOMMENDED)  |
+----------------+------------------+-------------------------+
| **CV**         | P-values         | Conservative p-value    |
|                | (KS, Chi²)       | stability               |
+----------------+------------------+-------------------------+
| **Plateau**    | Early detection, | Quick exploration,      |
|                | limited budget   | iterative tuning        |
+----------------+------------------+-------------------------+

**Recommended workflow**:

.. code-block:: python

    # Use Kneedle for RMSE (most reliable)
    results = adjuster.monte_carlo_fit(
        tests=['rmse'],
        stability_method='kneedle',
        smooth=True
    )
    
    recommended_n = results.attrs['recommended_size']
    print(f"RMSE stable at n = {recommended_n}")

Sampling strategies
-------------------

The helper `_generate_subsample_indices()` produces integer index lists that are
used by the Monte Carlo routines to build subsamples from the original dataset.
Below is a concise, practical explanation of the three supported strategies and
how to choose between them.

- random (without replacement)
  - What: Each subsample contains `size` unique indices drawn randomly from the
    original dataset (no duplicates inside the same subsample).
  - Constraint: `size` must be less than or equal to the original sample size
    `N`.
  - When to use: Typical default for studying how estimates change with smaller
    sample sizes. Keeps each subsample representative and avoids internal
    duplication.

- bootstrap (with replacement)
  - What: Each subsample is drawn with replacement, so the same original row
    can appear multiple times in a single subsample.
  - Constraint: None — `size` may be larger than `N` because indices can repeat.
  - When to use: Use when you need to estimate uncertainty (variance, CIs) via
    resampling, or when you want to allow `size` >= `N` for simulation purposes.

- disjoint (non-overlapping partitions)
  - What: The original indices are shuffled and partitioned into non-overlapping
    blocks of length `size`. Each block is a subsample with no shared indices.
  - Constraint: `size` must be <= `N`. The number of blocks per shuffle is
    `N // size`; for `n_repeats` larger than that, additional shuffles are used.
  - When to use: Use when you want independent partitions (similar to simple
    cross-validation) and want to avoid overlap between subsamples within the
    same shuffle.

Reproducibility and seed
------------------------

- A random `seed` controls the pseudo-random generator used to create indices.
  Providing the same seed reproduces the same sequence of subsamples. This is
  recommended for experiments where results must be repeatable.

Practical guidance
------------------

- Use `random` as a safe default when `size <= N` and you want unbiased
  subsamples without duplicates.
- Use `bootstrap` when you need to estimate uncertainty from resampling or when
  you want to allow `size >= N`.
- Use `disjoint` when you need non-overlapping partitions to compare independent
  fits or to maximize coverage of the original dataset without duplication.

See `_generate_subsample_indices()` source for exact behaviour and edge-case
handling.

Outputs
-------

- An `xarray.Dataset` with dimensions:
  - `sizes`: tested sample sizes
  - `repeats`: repetition index
- Data variables: `param_0, param_1, ...`, `ks_statistic`, `ks_pvalue`,
  `chi2_statistic`, `chi2_pvalue`, `rmse` (depending on `tests`).
- Attributes: 
  - `stability_points`: dict of detected sizes per variable with method info
  - `recommended_size`: sample size where primary metric stabilizes
  - `primary_metric`: the metric used for recommended_size (typically 'rmse')
  - `stable_pvalue_ks`, `stable_pvalue_chi2`, `stable_rmse`: metric values at stable point
  - `figure_path`: path to saved figure (if generated)
  - Metadata about sampling, bins, and configuration

Quick example
-------------

.. code-block:: python

    import numpy as np
    from magica.core import MagicAdjuster

    data = np.random.weibull(2, 1000)
    adjuster = MagicAdjuster(data)
    adjuster.fit_distribution('weibull_min')

    # Use Kneedle method for RMSE stability detection
    ds = adjuster.monte_carlo_fit(
        sizes=[50, 100, 200, 400, 600, 800],
        n_repeats=50,
        tests=['ks', 'chi2', 'rmse'],
        stability_method='kneedle',  # Recommended for RMSE
        smooth=True,                 # Smooth curves before detection
        sampling='random',
        fit_kwargs={'floc': 0},
        fig_output_path='mc_summary.png',
        plot_type='series'
    )

    # Access recommended size
    recommended_n = ds.attrs['recommended_size']
    print(f"Recommended sample size: {recommended_n}")
    
    # Check all stability points
    print(ds.attrs['stability_points'])

Interpretation tips
-------------------

- Look at medians/boxplots of parameters across `sizes` — convergence indicates
  stable estimation.
- Check `ks_pvalue` / `chi2_pvalue` behavior: rising p-values toward larger sizes
  suggest better fit at those sizes, but beware of the large-sample-size effect
  where p-values can become unreliable.
- **RMSE is the most reliable indicator** for stability: it shows smooth, monotonic
  decrease and clear convergence. Always include RMSE in your tests.
- Use `recommended_size` (based on primary metric, typically RMSE) as the
  recommended minimum sample size for robust parameter estimation.
- The generated figure shows:
  - Parameter convergence across sample sizes
  - Goodness-of-fit evolution (KS, Chi-square p-values, RMSE)
  - Red vertical lines marking detected stability points
  - Orange dashed lines showing smoothed curves (for Kneedle method)
  - Info boxes with metric values at stable points
- Compare stability points across different methods to understand metric behavior.

Method Comparison
-----------------

When analyzing your results, compare how different metrics stabilize:

.. code-block:: python

    stability = ds.attrs['stability_points']
    
    # RMSE typically stabilizes earliest and most reliably
    rmse_stable = stability['rmse']['size']
    rmse_method = stability['rmse']['method']
    
    # P-values may stabilize later or show erratic behavior
    ks_stable = stability['ks_pvalue']['size']
    
    print(f"RMSE stable at n = {rmse_stable} (method: {rmse_method})")
    print(f"KS p-value stable at n = {ks_stable}")
    
    # Use RMSE-based recommendation for production
    recommended = ds.attrs['recommended_size']
    print(f"\n⭐ Recommended sample size: {recommended}")

Further reading
---------------

See :doc:`/api/core` for function signature and options, and the example
notebook in this folder for a hands-on run.
