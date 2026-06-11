# Changelog

All notable changes to MagicA are documented here.

---

## [0.2.0] — 2026-06-11

### Breaking changes

#### (a) Fit API is now immutable

`fit_distribution()` (and the new canonical alias `fit()`) now returns an immutable
`FitResult` object instead of `self`.  Chaining on the returned value therefore
refers to `FitResult`, not to `DataProcessor` or `MagicAdjuster`.

```python
# Before
fitted = data.fit_distribution('weibull')
params = data.get_fitted_params()
cdf_vals = data.cdf(x)

# After
fit = data.fit('weibull')
params   = fit.params
cdf_vals = fit.cdf(x)
```

`DataProcessor.__getattr__` delegation and the mutable `_adjuster` cache are
removed.  `get_fitted_params()` and `get_distribution_info()` are removed from
`DataProcessor` (use `fit.params` and `fit.info` on the returned `FitResult`).

Two fits on the same processor no longer overwrite each other (the shared-state
bug documented in the previous TO-DO is resolved by construction).

`FitResult.data` holds a **shared reference** to the underlying array — no copy
is ever made for a fit result.

See [MIGRATION.md](MIGRATION.md) for full before/after syntax.

#### (b) PoT return levels are numerically different (bug fix)

PoT + GPD return values computed by `ExtremesAnalyzer` will differ from 0.1.x.
This is intentional: the old code omitted the exceedance rate λ from the formula
and fit on the full time series instead of the extracted exceedances.

New formula (correct):

```
x_T = u + (σ/ξ)·((λ·T)^ξ − 1)    if |ξ| ≥ 1e-6
x_T = u + σ·ln(λ·T)               if |ξ| < 1e-6
```

where `u` is the threshold and `λ = n_independent / time_span`.

Block Maxima + GEV return values are unchanged.

#### (c) EVA families are now restricted

`fit_block_maxima()` and `fit_pot()` only accept theoretically justified families:

```python
EVA_FAMILIES = {
    "bm":  ["genextreme", "gumbel_r"],
    "pot": ["genpareto", "expon"],
}
```

Passing an unsupported distribution raises a clear `ValueError`.

### New features

- **`FitResult`** immutable dataclass with `pdf`, `cdf`, `sf`, `logpdf`, `logcdf`,
  `logsf` (smart defaults to original data when `x=None`), `ppf`, `isf`, `rvs`,
  `stats`, `frozen`, `info`, and `goodness_of_fit()`.
- **`EVAFit`** (extends `FitResult`) carries EVA metadata:
  - `method`: `'bm'` or `'pot'`
  - `threshold`, `lambda_rate`, `blocks_per_year`
  - `return_value(T)`, `return_period(x)` with method-dispatched formulas
  - `return_level_table(periods)` → serialisable list of dicts for frontends
    (period, level, ci\_low=None, ci\_high=None)
- **`ExtremesAnalyzer.fit_block_maxima(block_maxima, ...)`** — fits on extracted sample
- **`ExtremesAnalyzer.fit_pot(pot_result)`** — fits GPD to `exceedances - u`, `floc=0`
- **`EVA_FAMILIES`** registry (importable from `magica.core.auto_fitter`)
- **`AutoFitter.fit_best_distribution()`** now returns `FitResult` (not a dict)
- All `FitResult` candidates in `AutoFitter` share the same data array; only
  scalar metrics per candidate are retained in memory.

### Removals / deprecations

- `DataProcessor.get_fitted_params()` — removed. Use `fit.params`.
- `DataProcessor.get_distribution_info()` — removed. Use `fit.info`.
- `DataProcessor.__getattr__` delegation to internal adjuster — removed.
- `AutoFitter.get_best_adjuster()` — deprecated; use `fit_best_distribution()`.
- `ExtremesAnalyzer.fit_distribution()` — kept for backward compatibility (BM path
  only); prefer `fit_block_maxima()` or `fit_pot()`.

### Internal / maintenance

- Matplotlib imports are lazy (inside functions) throughout the package.
- `xarray` import is lazy (inside `monte_carlo_fit`).
- Deprecated pandas frequency aliases `'A'` → `'YE'` and `'H'` → `'h'` corrected
  in all defaults and docstrings.

---

## [0.1.0] — initial release

First public release of MagicA featuring:
- `DataProcessor`: data loading and statistics
- `MagicAdjuster`: distribution fitting with goodness-of-fit tests
- `AutoFitter`: automatic distribution selection
- `ExtremesAnalyzer`: block maxima and POT extraction, return-level plots
- Monte Carlo stability analysis (`monte_carlo_fit`)
- Synthetic wind data generators
