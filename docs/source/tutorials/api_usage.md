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
params = fitted.get_fitted_params()

# 3. Evaluate goodness-of-fit
chi2 = fitted.goodness_of_fit('chi2')
ks = fitted.goodness_of_fit('ks')
rms = fitted.goodness_of_fit('rms')

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
size_200_data = results.sel(sizes=200)
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

### `DataProcessor.fit_distribution(distribution)`
Fits a statistical distribution to the data.
- **Parameters:**
  - `distribution`: str or scipy.stats distribution (e.g., 'weibull', 'norm', 'gamma')
- **Returns:** `DataProcessor` (with fitted distribution)

### `DataProcessor.get_fitted_params()`
Returns fitted parameters of the distribution.
- **Returns:** tuple

### `DataProcessor.goodness_of_fit(method, bins='doane')`
Evaluates the fit using a statistical test.
- **Parameters:**
  - `method`: 'chi2', 'ks', or 'rms'
  - `bins`: binning method (default 'doane')
- **Returns:** dict with test results

### `MagicAdjuster.monte_carlo_fit(sizes=None, n_repeats=20, tests=['ks'], fig_output_path=None, plot_type='series', sampling='random', distribution_params=None, **kwargs)`
Performs Monte Carlo stability analysis to determine minimum sample size for reliable parameter estimation.
- **Parameters:**
  - `sizes`: list, optional - Explicit list of sample sizes
  - `n_repeats`: int - Repetitions per size (default 20)
  - `tests`: list - GOF tests to perform ['chi2','ks','rmse']
  - `fig_output_path`: str, optional - Save 2x3 summary figure if provided (includes red dashed stability lines)
  - `plot_type`: 'series' or 'boxplots' - Panel style when saving figure
  - `sampling`: 'random','bootstrap','disjoint' - subsampling strategy
  - `distribution_params`: tuple, optional - Use fixed params (no refitting)
  - `bins`: (kwarg) binning method for chi2 (default 'doane')
  - `fit_kwargs`: (kwarg) constraints passed to fit_distribution
  - `plot`: str or bool - Plotting option ('series', 'boxplots', True, or False) (default True)
  - `sampling`: str - Sampling strategy ('random' or 'bootstrap') (default 'random')
  - `distribution_params`: tuple, optional - Pre-calculated distribution parameters
  - `bins`: str or int - Binning strategy for chi-square test (default 'doane')
  - `fit_kwargs`: dict - Additional arguments passed to fit_distribution()
- **Returns:** xarray.Dataset with dimensions ['sizes', 'repeats'] and variables for parameters and test results
  - `method`: 'chi2', 'ks', or 'rms'
  - `bins`: binning method (default 'doane')
- **Returns:** dict with test results

### `DataProcessor.get_basic_stats()`
Returns basic statistics of the data.
- **Returns:** dict

### `DataProcessor.get_distribution_info()`
Returns info about the fitted distribution.
- **Returns:** dict

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
size_200 = results.sel(sizes=200)
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
figure = results.attrs['figure']
```

## Goodness-of-Fit Methods
- `'chi2'`: Chi-square test
- `'ks'`: Kolmogorov-Smirnov test
- `'rms'`: Root Mean Square error between observed and estimated PDF

---
See also: [example_magic_adjuster.py](example_magic_adjuster.py)
