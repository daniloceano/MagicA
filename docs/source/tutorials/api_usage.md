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

### `DataProcessor.get_basic_stats()`
Returns basic statistics of the data.
- **Returns:** dict

### `DataProcessor.get_distribution_info()`
Returns info about the fitted distribution.
- **Returns:** dict

## Goodness-of-Fit Methods
- `'chi2'`: Chi-square test
- `'ks'`: Kolmogorov-Smirnov test
- `'rms'`: Root Mean Square error between observed and estimated PDF

---
See also: [example_magic_adjuster.py](example_magic_adjuster.py)
