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
- `sampling`: sampling strategy (`random`, `bootstrap`, `disjoint`).
- `fit_kwargs`: passed to the underlying fit routine (e.g., `floc=0`).
- `distribution_params`: optional fixed parameters (skip fitting when supplied).
- `fig_output_path`: optional path to save the 2x3 summary figure.
- `plot_type`: `'series'` (median + IQR) or `'boxplots'`.

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
- Attributes: `stability_points` (dict of detected sizes per variable),
  `figure_path` (if a figure was saved), and metadata about sampling and bins.

Quick example
-------------

.. code-block:: python

    import numpy as np
    from magica.core import MagicAdjuster

    data = np.random.weibull(2, 1000)
    adjuster = MagicAdjuster(data)
    adjuster.fit_distribution('weibull_min')

    ds = adjuster.monte_carlo_fit(
        sizes=[50,100,200,400],
        n_repeats=50,
        tests=['ks','chi2'],
        sampling='random',
        fit_kwargs={'floc': 0},
        fig_output_path='mc_summary.png',
        plot_type='series'
    )

    print(ds.attrs['stability_points'])

Interpretation tips
-------------------

- Look at medians/boxplots of parameters across `sizes` — convergence indicates
  stable estimation.
- Check `ks_pvalue` / `chi2_pvalue` behavior: rising p-values toward larger sizes
  suggest better fit at those sizes.
- Use `stability_points` as a conservative recommendation for minimum sample size.

Further reading
---------------

See :doc:`/api/core` for function signature and options, and the example
notebook in this folder for a hands-on run.
