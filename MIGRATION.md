# MagicA Migration Guide: 0.1.x → 0.2.0

## Summary of breaking changes

| Area | Before (0.1.x) | After (0.2.0) |
|---|---|---|
| Fit return type | `fit_distribution()` returned `self` (DataProcessor or MagicAdjuster) | `fit()` / `fit_distribution()` returns immutable `FitResult` |
| Fit state | Second fit overwrote the first | Each fit is independent; both coexist |
| PDF/CDF/PPF | Called on `DataProcessor` or `MagicAdjuster` | Called on `FitResult` |
| `get_fitted_params()` | Method on `DataProcessor` | `fit_result.params` |
| `get_distribution_info()` | Method on `DataProcessor` | `fit_result.info` |
| `get_best_adjuster()` | Returned `MagicAdjuster` | Returns `FitResult` via `fit_best_distribution()` |
| PoT return levels | Numerically wrong (λ ignored) | Correct formula with λ |
| EVA fit target | Full time series | Extracted extreme sample only |
| EVA families | Any distribution | Restricted: BM (genextreme, gumbel_r), PoT (genpareto, expon) |

---

## 1. Distribution fitting

### Before

```python
# fit_distribution mutated state and returned self
fitted = data.fit_distribution('weibull', floc=0)  # returns DataProcessor
params = data.get_fitted_params()
info   = data.get_distribution_info()
cdf_vals = data.cdf(x)   # delegated via __getattr__
```

### After

```python
fit = data.fit('weibull', floc=0)   # returns FitResult
params   = fit.params               # tuple
info     = fit.info                 # dict: name, parameters, num_params, data_size
cdf_vals = fit.cdf(x)              # on the FitResult
pdf_vals = fit.pdf()               # defaults to original data when x is None
ppf_99   = fit.ppf(0.99)
ks_test  = fit.goodness_of_fit('ks')
rmse     = fit.goodness_of_fit('rmse')
```

`fit_distribution` still works as an alias for `fit` with the same new semantics.

### Two independent fits (previously broken, now works)

```python
# Before: fit_gamma would overwrite fit_weibull's state
fit_w = data.fit('weibull', floc=0)
fit_g = data.fit('gamma')
# Both fit objects are independent and hold their own params:
fit_w.name   # 'weibull'
fit_g.name   # 'gamma'
fit_w.ppf(0.99)  # uses weibull params
fit_g.ppf(0.99)  # uses gamma params
```

---

## 2. AutoFitter

### Before

```python
auto   = data.get_auto_fitter()
result = auto.fit_best_distribution()   # returned a dict
adjuster = auto.get_best_adjuster()     # returned MagicAdjuster
```

### After

```python
auto = data.get_auto_fitter()
fit  = auto.fit_best_distribution()   # returns FitResult
# Use it directly:
fit.name
fit.params
fit.goodness_of_fit('ks')
# get_best_adjuster() still works but emits DeprecationWarning
```

---

## 3. Monte Carlo stability analysis

The call order changes slightly because `fit_distribution` no longer returns `self` (an adjuster):

### Before

```python
adjuster = data.fit_distribution('weibull', floc=0)   # returned MagicAdjuster
results = adjuster.monte_carlo_fit(tests=['ks', 'rmse'])
```

### After

```python
fit      = data.fit('weibull', floc=0)         # FitResult
adjuster = data._get_adjuster()                 # MagicAdjuster (stateless)
adjuster.fit_distribution('weibull', floc=0)    # sets internal distribution for MC
results  = adjuster.monte_carlo_fit(tests=['ks', 'rmse'])
```

Or more concisely:

```python
adjuster = data._get_adjuster()
fit = adjuster.fit_distribution('weibull', floc=0)   # both gets FitResult and sets MC state
results = adjuster.monte_carlo_fit(tests=['ks', 'rmse'])
```

---

## 4. Extreme value analysis

### Before (broken)

```python
extremes = processor.get_extremes_analyzer()
extremes.fit_distribution('genextreme')  # fit on FULL series (bug)
rv = extremes.return_value(100)          # wrong formula for PoT (no λ)
```

### After (Block Maxima)

```python
extremes    = processor.get_extremes_analyzer()
annual_max, times = extremes.extract_block_maxima('YE')
ev_fit      = extremes.fit_block_maxima(annual_max, times)   # fits on extracted maxima
rv_100      = ev_fit.return_value(100)      # correct BM formula
table       = ev_fit.return_level_table([10, 50, 100])  # serialisable for frontend
```

### After (Peaks over Threshold)

```python
extremes = processor.get_extremes_analyzer()
result   = extremes.find_optimal_pot_threshold(min_samples=50)
ev_fit   = extremes.fit_pot(result)      # fits genpareto to exceedances - u, floc=0
rv_100   = ev_fit.return_value(100)      # correct formula: u + (σ/ξ)·((λ·T)^ξ − 1)
table    = ev_fit.return_level_table([10, 50, 100])
```

### PoT numerical change

Return levels computed via PoT + GPD **will differ numerically** from 0.1.x because:

1. The fit is now on `exceedances - u` (exceedances above threshold) with `floc=0`, not on the full dataset.
2. The return-level formula now accounts for the exceedance rate λ.  
   Old formula (wrong): `ppf(1 - 1/T)`  
   New formula (correct): `u + (σ/ξ)·((λ·T)^ξ − 1)` where `λ = n_independent / time_span`.

When `λ ≈ 1` the difference is small; for typical datasets with many exceedances per year (e.g. `λ = 5`) the error in the old formula can exceed several metres per second for long return periods.

### Restricted EVA families

Passing an unsupported distribution now raises a clear `ValueError`:

```python
# Before: silently accepted any distribution
extremes.fit_block_maxima(annual_max, distribution='weibull_min')  # now raises ValueError
```

Allowed families:

```python
from magica.core.auto_fitter import EVA_FAMILIES
# {'bm': ['genextreme', 'gumbel_r'], 'pot': ['genpareto', 'expon']}
```

---

## 5. FitResult data reference

`FitResult.data` is a **shared reference** to the original numpy array — no copy is made.
This matters for memory-efficient batch/grid processing:

```python
fit = data.fit('weibull')
assert fit.data is data.data   # True — same object, no copy
```
