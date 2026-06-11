# MagicA Core Module — Architecture Reference

## Overview

The `core` module provides MagicA's statistical analysis engine via four classes:

| Class | Role |
|---|---|
| `DataProcessor` | Data loading, validation, and entry point for all analyses |
| `FitResult` | Immutable result of a single distribution fit |
| `MagicAdjuster` | Stateless fitter; also runs Monte Carlo stability analysis |
| `AutoFitter` | Automatic model selection across multiple candidates |
| `ExtremesAnalyzer` | Extreme value analysis (Block Maxima and PoT) |
| `EVAFit` | Immutable EVA result with correct return-level formulas |

---

## Quick start

```python
import magica as ma
import numpy as np

data = ma.read_data(np.random.weibull(2, 1000) * 8)

# Standard fitting
fit = data.fit('weibull', floc=0)   # returns FitResult
fit.params                          # tuple of fitted parameters
fit.cdf(np.array([5., 10.]))        # CDF at custom points
fit.pdf()                           # PDF evaluated at the original data
fit.ppf(0.99)                       # 99th-percentile quantile
fit.goodness_of_fit('ks')           # Kolmogorov-Smirnov test

# Two independent fits coexist without interference
fit_w = data.fit('weibull', floc=0)
fit_g = data.fit('gamma')
fit_w.name   # 'weibull'
fit_g.name   # 'gamma'
```

---

## FitResult — immutable fit container

`FitResult` is a frozen dataclass.  Each call to `fit()` returns a fresh,
independent instance; no state from a previous call is retained.

```python
@dataclass(frozen=True, eq=False)
class FitResult:
    distribution: object   # scipy dist object (unfrozen)
    name: str
    params: tuple          # fitted parameters
    data: np.ndarray       # shared reference — no copy
```

### Methods with smart defaults

`pdf`, `cdf`, `sf`, `logpdf`, `logcdf`, `logsf` — when called with no argument,
evaluate at the original data:

```python
fit.cdf()          # same as fit.cdf(fit.data)
fit.cdf(x)         # evaluate at custom points
fit.ppf(0.99)      # percent-point function (explicit argument required)
fit.goodness_of_fit('rmse')
fit.goodness_of_fit('ks')
fit.goodness_of_fit('aic')
fit.info            # {'name': ..., 'parameters': ..., 'num_params': ..., 'data_size': ...}
```

### Memory note

`FitResult.data` is a **shared reference** — no copy is made.  All fits on
the same `DataProcessor` point to the same underlying array, so the marginal
cost of each `FitResult` is just a name string + parameter tuple (order of a
few hundred bytes).

---

## AutoFitter — automatic model selection

Used only for the **standard statistics path**.  Do not use for extreme value
analysis; use `ExtremesAnalyzer` instead.

```python
auto = data.get_auto_fitter(criterion='rmse')
best = auto.fit_best_distribution()   # returns FitResult
best.name, best.goodness_of_fit('rmse')

# Comparison table (scalars only, no heavy objects)
table = auto.get_comparison_table(sort_by='aic')
```

### Memory-efficient batch pattern

```python
results = []
for point_data in grid_points:
    proc = ma.read_data(point_data)
    auto = proc.get_auto_fitter()
    best = auto.fit_best_distribution()
    results.append({'name': best.name, 'params': best.params,
                    'rmse': best.goodness_of_fit('rmse')})
    # best goes out of scope; heavy objects freed
```

### EVA_FAMILIES — restricted distributions for extremes

```python
from magica.core.auto_fitter import EVA_FAMILIES
# {'bm': ['genextreme', 'gumbel_r'], 'pot': ['genpareto', 'expon']}
```

---

## MagicAdjuster — stateless fitter and Monte Carlo engine

`MagicAdjuster` is used internally by `DataProcessor.fit()`.  Users generally
interact with it only when running Monte Carlo stability analysis.

```python
adjuster = data._get_adjuster()
fit = adjuster.fit_distribution('weibull', floc=0)  # FitResult
results = adjuster.monte_carlo_fit(tests=['ks', 'rmse'], stability_method='kneedle')
print(results.attrs['recommended_size'])
```

`fit_distribution()` returns `FitResult` and retains the distribution name
internally so that `monte_carlo_fit()` knows which distribution to resample.

---

## ExtremesAnalyzer — Block Maxima and Peaks over Threshold

### Block Maxima (GEV)

```python
extremes = processor.get_extremes_analyzer()

# 1. Extract annual maxima
annual_max, times = extremes.extract_block_maxima('YE')

# 2. Fit GEV to the extracted sample (not the full series)
ev_fit = extremes.fit_block_maxima(annual_max, times, distribution='genextreme')

# 3. Return levels
rv_100 = ev_fit.return_value(100)            # 100-year return value
rp     = ev_fit.return_period(35.0)          # return period for 35 m/s
table  = ev_fit.return_level_table([10, 50, 100])  # serialisable for frontends
```

Allowed BM distributions: `'genextreme'`, `'gumbel_r'`.

### Peaks over Threshold (GPD)

```python
# 1. Find threshold
result = extremes.find_optimal_pot_threshold(min_samples=50)

# 2. Fit GPD to exceedances above threshold
ev_fit = extremes.fit_pot(result, distribution='genpareto')
# Internally: fits to (exceedances - u) with floc=0, giving params (ξ, 0, σ)

# 3. Return levels use correct formula: x_T = u + (σ/ξ)·((λ·T)^ξ − 1)
rv_100 = ev_fit.return_value(100)
```

Allowed PoT distributions: `'genpareto'`, `'expon'`.

### EVAFit attributes

```python
ev_fit.method          # 'bm' or 'pot'
ev_fit.threshold       # u (PoT only)
ev_fit.lambda_rate     # λ = n_independent / time_span (PoT only)
ev_fit.blocks_per_year # (BM only)
ev_fit.params          # fitted distribution parameters
```

### Return-level formulas

**Block Maxima:**

```
x_T = ppf(1 - 1/T)
T   = 1 / (1 - CDF(x))
```

**PoT + GPD** (convention: fit to exceedances - u, floc=0 → params = (ξ, 0, σ)):

```
x_T = u + (σ/ξ)·((λ·T)^ξ − 1)   if |ξ| ≥ 1e-6
x_T = u + σ·ln(λ·T)              if |ξ| < 1e-6

λ_x = λ·(1 + ξ·(x−u)/σ)^(−1/ξ)  if |ξ| ≥ 1e-6
λ_x = λ·exp(−(x−u)/σ)            if |ξ| < 1e-6
T   = 1 / λ_x
```

---

## Deprecated / removed API

| Old | Replacement |
|---|---|
| `data.fit_distribution('w').cdf(x)` (chaining on DataProcessor) | `data.fit('w').cdf(x)` |
| `data.get_fitted_params()` | `fit.params` |
| `data.get_distribution_info()` | `fit.info` |
| `data.cdf(x)` (delegated via `__getattr__`) | `fit.cdf(x)` |
| `auto.get_best_adjuster()` | `auto.fit_best_distribution()` → FitResult |
| `extremes.fit_distribution('genextreme')` | `extremes.fit_block_maxima(maxima, ...)` |
