# MagicA API Usage

## Basic Example: Weibull Fit and Goodness-of-Fit Evaluation

```python
import magica as ma
import numpy as np

# Example wind speed data
wind_data = [2.1, 5.4, 8.7, 12.3, 6.8, 9.1, 15.2, 3.4, 7.6, 11.0, 4.5, 13.2, 8.9, 6.7, 10.5]

# 1. Load data
processor = ma.read_data(wind_data)

# 2. Fit Weibull distribution
fitted = processor.fit_distribution('weibull')
params = fitted.params

# 3. Evaluate goodness-of-fit
chi2 = fitted.goodness_of_fit('chi2')
ks = fitted.goodness_of_fit('ks')
rms = fitted.goodness_of_fit('rmse')

print('Weibull parameters:', params)
print('Chi-square:', chi2)
print('Kolmogorov-Smirnov:', ks)
print('Root Mean Square:', rms)
```

## Monte Carlo Stability Analysis

Determine the minimum sample size needed for stable parameter estimation:

```python
import numpy as np
from magica.core import MagicAdjuster

# Generate larger dataset for Monte Carlo analysis
data = np.random.weibull(2, 1000)

# Create adjuster and fit distribution
adjuster = MagicAdjuster(data)
adjuster.fit_distribution('weibull_min')

# Run Monte Carlo stability analysis
results = adjuster.monte_carlo_fit(
  n_repeats=100,
  tests=['chi2', 'ks']  # no figure by default
)

# Work with xarray Dataset
ks_pvalues = results['ks_pvalue']
param_medians = results['param_0'].median(dim='repeats')

# Check stability points
stability = results.attrs['stability_points']
print(f"Parameter 0 stabilizes at: {stability['param_0']['size']}")
```

### Advanced Monte Carlo Usage

```python
# Custom sizes with fitting constraints
results = adjuster.monte_carlo_fit(
  sizes=[50, 100, 200, 400],
  n_repeats=200,
  tests=['chi2', 'ks', 'rmse'],
  plot_type='boxplots',
  fig_output_path='stability_boxplots.png',
  bins='scott',
  fit_kwargs={'floc': 0}
)

# Using pre-calculated parameters (bypass fitting)
known_params = (2.0, 0.0, 1.0)  # shape, loc, scale for Weibull
results = adjuster.monte_carlo_fit(
    distribution_params=known_params,
    n_repeats=150,
    tests=['chi2', 'ks']
)

# Easy data manipulation with xarray
size_200_data = results.sel(sizes=200, method='nearest')
param_convergence = results['param_0'].std(dim='repeats')

# Visualize parameter stability
results['param_0'].plot(x='sizes', hue='repeats', alpha=0.3)
```

## API Reference

### `read_data(data)`
Loads and validates data for analysis.
- **Parameters:**
  - `data`: array-like (list, numpy array, pandas Series/DataFrame)
- **Returns:** `DataProcessor` instance

### `DataProcessor.fit(distribution, **kwargs)`
Fits a distribution and returns an immutable `FitResult`.
`fit_distribution()` remains an alias with the same return type.

### `FitResult.params` and `FitResult.info`
The fitted parameter tuple and a dictionary describing the fit.

### `FitResult.goodness_of_fit(method, **kwargs)`
Accepts `'chi2'`, `'ks'`, `'rmse'`, `'aic'`, and `'bic'`.
Chi-square and KS return dictionaries; RMSE, AIC, and BIC return floats.
`bins` controls chi-square binning (default `'doane'`).

### `MagicAdjuster.monte_carlo_fit(...)`
Returns an `xarray.Dataset` with `sizes` and `repeats` dimensions.
Fit the distribution on the same adjuster before calling this method.
Defaults include `n_repeats=20`, `tests=['ks']`, `stability_method='kneedle'`,
`sampling='random'`, `plot_type='series'`, and `fig_output_path=None`.
`fit_kwargs` passes fitting constraints; `distribution_params` bypasses refitting.
See the [Monte Carlo API reference](../api/monte_carlo.rst) for the full signature.

### `DataProcessor.get_basic_stats()`
Returns a dictionary of descriptive statistics.

## Working with xarray Results

The `monte_carlo_fit` method returns results as an xarray Dataset, providing powerful data manipulation capabilities:

```python
# Run Monte Carlo analysis
results = adjuster.monte_carlo_fit(n_repeats=50, tests=['ks', 'chi2'])

# Dataset overview
print(results)
print(results.dims)  # Dimensions: sizes, repeats
print(results.data_vars)  # Variables: param_0, ks_statistic, ks_pvalue, etc.

# Select data for specific sample size
size_200 = results.sel(sizes=200, method='nearest')
print(size_200['param_0'].values)  # All parameter values for size 200

# Calculate statistics across repeats
param_means = results['param_0'].mean(dim='repeats')
param_stds = results['param_0'].std(dim='repeats')
ks_medians = results['ks_pvalue'].median(dim='repeats')

# Select size range
large_sizes = results.sel(sizes=slice(200, None))

# Convert to pandas DataFrame for further analysis
df = results.to_dataframe()

# Plot with built-in xarray methods
results['ks_pvalue'].plot(x='sizes', hue='repeats', alpha=0.5)

# Access metadata and stability points
stability = results.attrs['stability_points']
figure_path = results.attrs['figure_path']
```

## Goodness-of-Fit Methods
- `'chi2'`: Chi-square test
- `'ks'`: Kolmogorov-Smirnov test
- `'rms'`: Root Mean Square error between observed and estimated PDF

---
See also: [MagicAdjuster tutorial](magic_adjuster_tutorial.ipynb)
