# MagicA Core Module - Understanding the Architecture

## 🎯 What This Module Does

The `core` module contains the main classes that power MagicA's statistical analysis capabilities. Think of it as the "brain" of the package where data processing and statistical fitting happen.

## 📋 Quick Overview

This module has three main classes that work together:
- **`DataProcessor`**: Handles data loading, cleaning, and basic statistics
- **`MagicAdjuster`**: Performs advanced statistical distribution fitting
- **`AutoFitter`**: Automatically tests multiple distributions and selects the best fit

The clever part? You only need to work with `DataProcessor` - it automatically creates and manages the other classes when needed!

## 🏗️ The Architecture: How They Work Together

### The Problem We Solved

In typical programming, you might need to do this:
```python
# Traditional approach (what we DON'T want)
data = load_data("wind_speed.csv")
processor = DataProcessor(data)
adjuster = MagicAdjuster(processor.get_data())  # Manual creation
adjuster.fit_distribution('weibull')
```

This means:
- Creating multiple objects manually
- Remembering which object does what
- Managing relationships between objects

### Our Solution: Smart Integration

Instead, with MagicA you simply do:
```python
# MagicA approach (what we DO want)
processor = ma.read_data("wind_speed.csv")
processor.fit_distribution('weibull')  # Automatically handles everything!
```

## 🔧 How It Works: The Magic Behind the Scenes

### 1. Lazy Initialization Pattern

**What it means**: Create objects only when you actually need them, not before.

**Why it matters**: Saves memory and processing time.

**Real-world analogy**: Like having a toolbox in your garage. You don't take out every tool when you enter the garage - you only grab the hammer when you need to hammer something.

```python
class DataProcessor:
    def __init__(self, data=None):
        self.data = None
        self._adjuster = None  # 🎯 Not created yet! Just a placeholder
        
        if data is not None:
            self.load_data(data)
```

### 2. Factory Method Pattern

**What it means**: A special method that creates objects for you when needed.

**Why it matters**: You don't need to remember how to create complex objects.

**Real-world analogy**: Like calling a restaurant. You don't need to know how to cook - you just order, and they make it for you.

```python
def _get_adjuster(self):
    """Factory method - creates the adjuster when first needed"""
    if self._adjuster is None:  # 🔍 Check: Do we have one already?
        from .magic_adjuster import MagicAdjuster
        self._adjuster = MagicAdjuster(self)  # 🏭 Create it now!
    return self._adjuster  # 🎁 Return the adjuster (new or existing)
```

### 3. Delegation Pattern

**What it means**: One object passes work to another object that specializes in that task.

**Why it matters**: Each class focuses on what it does best.

**Real-world analogy**: Like a project manager (DataProcessor) who delegates specialized tasks to experts (MagicAdjuster) while maintaining overall control.

```python
def fit_distribution(self, distribution, **kwargs):
    """DataProcessor delegates fitting work to MagicAdjuster"""
    adjuster = self._get_adjuster()  # 🏭 Get/create the specialist
    adjuster.fit_distribution(distribution, **kwargs)  # 🎯 Delegate the work
    return self  # 🔄 Return self for method chaining
```

## 📊 Step-by-Step Example: Wind Speed Analysis

Let's follow a complete example to see how everything works together:

### Step 1: Loading Data
```python
import magica as ma
import numpy as np

# Simulate some wind speed data (m/s)
wind_speeds = np.random.weibull(2, 1000) * 8 + 2

# Load into MagicA
processor = ma.read_data(wind_speeds)
print(processor)
# Output: DataProcessor(length=1000, dtype=float64)
```

**What happened internally**:
- ✅ Data loaded and cleaned
- ❌ No `_adjuster` created yet (still `None`)

### Step 2: Basic Statistics (No MagicAdjuster Needed)
```python
stats = processor.get_basic_stats()
print(f"Mean wind speed: {stats['mean']:.2f} m/s")
print(f"Standard deviation: {stats['std']:.2f} m/s")
```

**What happened internally**:
- ✅ Basic statistics calculated by `DataProcessor`
- ❌ Still no `_adjuster` created (not needed for basic stats)

### Step 3: Distribution Fitting (MagicAdjuster Created Automatically)
```python
# This is where the magic happens!
processor.fit_distribution('weibull')
print(processor)
# Output: DataProcessor(length=1000, dtype=float64, distribution=weibull_min)
```

**What happened internally**:
1. 🔍 `fit_distribution()` calls `_get_adjuster()`
2. 🏭 `_get_adjuster()` sees `_adjuster` is `None`, so creates `MagicAdjuster(self)`
3. 🎯 Work is delegated to the new adjuster
4. ✅ Distribution fitted and parameters stored

### Step 4: Getting Results (Using Existing MagicAdjuster)
```python
params = processor.get_fitted_params()
print(f"Weibull parameters: {params}")

info = processor.get_distribution_info()
print(f"Distribution: {info['name']}")
print(f"AIC: {info['aic']:.2f}")
```

**What happened internally**:
- ✅ Uses the existing `_adjuster` (no new creation)
- 🎯 Results retrieved from stored fit

## 🎨 Method Chaining: Fluent Interface

Because methods return `self`, you can chain operations:

```python
# All in one fluent chain!
result = (ma.read_data(wind_speeds)
           .fit_distribution('weibull')
           .get_fitted_params())

print(result)
```

This is like saying: "Load data, then fit Weibull, then give me parameters" in one smooth sentence.

## 🧠 Memory and Performance Benefits

### Memory Efficiency
```python
# Create processor
processor = ma.read_data(small_dataset)
# Memory usage: ~X MB (just data + metadata)

# Only do basic stats - no extra memory used
stats = processor.get_basic_stats()
# Memory usage: still ~X MB

# Fit distribution - now MagicAdjuster is created
processor.fit_distribution('weibull')
# Memory usage: ~X + Y MB (data + adjuster)
```

### When Objects Are Created
| Operation | DataProcessor | MagicAdjuster |
|-----------|---------------|---------------|
| `ma.read_data()` | ✅ Created | ❌ Not created |
| `get_basic_stats()` | ✅ Used | ❌ Still not created |
| `fit_distribution()` | ✅ Used | ✅ **Now created** |
| `get_fitted_params()` | ✅ Used | ✅ Reused |

## 🔍 Common Questions

### Q: "Why not create MagicAdjuster immediately?"
**A**: Because you might only need basic statistics! Why use extra memory and processing time for features you don't need?

### Q: "What if I call fit_distribution() multiple times?"
**A**: The same `MagicAdjuster` instance is reused. It's created once and then reused for efficiency.

### Q: "Can I access the MagicAdjuster directly?"
**A**: You shouldn't need to! The `DataProcessor` provides all the methods you need. This is called "encapsulation" - hiding complex details behind a simple interface.

### Q: "What happens if fitting fails?"
**A**: The `MagicAdjuster` handles errors gracefully and provides informative error messages.

## 🚀 Advanced Example: Multiple Distributions

```python
import magica as ma
import numpy as np

# Load data
data = ma.read_data(np.random.exponential(5, 1000))

# Try different distributions
distributions = ['exponential', 'gamma', 'weibull', 'lognorm']
results = {}

for dist in distributions:
    # Each fit_distribution call uses the SAME MagicAdjuster instance
    data.fit_distribution(dist)
    info = data.get_distribution_info()
    results[dist] = info['aic']  # Akaike Information Criterion

# Find best distribution
best_dist = min(results, key=results.get)
print(f"Best distribution: {best_dist} (AIC: {results[best_dist]:.2f})")
```

## 📚 Key Takeaways

1. **Simplicity**: You work with one main object (`DataProcessor`)
2. **Efficiency**: Complex objects are created only when needed
3. **Flexibility**: The system adapts to your workflow
4. **Reliability**: Error handling and state management are built-in
5. **Performance**: Memory and computation are optimized automatically

## 🤖 Automatic Distribution Fitting with AutoFitter

### The New Challenge: Model Selection

While `MagicAdjuster` excels at fitting a specific distribution, real-world data analysis often requires testing multiple distributions to find the best fit. This is where `AutoFitter` comes in.

### AutoFitter Architecture: Smart Model Selection

**Problem**: Testing multiple distributions manually:
```python
# Manual approach (tedious)
processor = ma.read_data(data)
processor.fit_distribution('weibull')
weibull_rmse = processor.goodness_of_fit('rmse')

processor.fit_distribution('gamma') 
gamma_rmse = processor.goodness_of_fit('rmse')

processor.fit_distribution('lognorm')
lognorm_rmse = processor.goodness_of_fit('rmse')
# ... and compare manually
```

**Solution**: Automatic testing and selection:
```python
# AutoFitter approach (elegant)
processor = ma.read_data(data)
auto_fitter = processor.get_auto_fitter()
best_result = auto_fitter.fit_best_distribution()
print(f"Best distribution: {best_result['distribution']}")
```

### Lazy Initialization in AutoFitter

`AutoFitter` uses advanced lazy initialization - it creates separate `MagicAdjuster` instances only when testing each distribution:

```python
class AutoFitter:
    def __init__(self, data_processor, candidates=None):
        # Placeholders created, but NO MagicAdjuster instances yet!
        self._adjusters = {dist: None for dist in self.candidates}
    
    def _get_adjuster(self, distribution):
        # Factory method - creates adjuster only when first needed
        if self._adjusters[distribution] is None:
            temp_processor = DataProcessor()
            temp_processor.data = self.data_processor.data.copy()
            self._adjusters[distribution] = MagicAdjuster(temp_processor)
        return self._adjusters[distribution]
```

### Memory Efficiency Benefits

| Operation | Memory Usage | MagicAdjuster Instances |
|-----------|--------------|-------------------------|
| `get_auto_fitter()` | Base + placeholders | 0 created |
| `fit_single_distribution('weibull')` | Base + 1 adjuster | 1 created |
| `fit_all_distributions()` | Base + N adjusters | N created (as needed) |

### Complete AutoFitter Example

```python
import magica as ma
import numpy as np

# Generate complex wind speed data
wind_data = np.concatenate([
    np.random.weibull(2, 500) * 8 + 2,    # Low wind period
    np.random.lognormal(2, 0.5, 300),     # Variable wind period  
    np.random.gamma(3, 2, 200)            # High wind period
])

# Load data
processor = ma.read_data(wind_data)

# Show all available distributions (113 total!)
from magica.core.auto_fitter import AutoFitter
all_distributions = AutoFitter.get_all_available_distributions()
print(f"Available distributions: {len(all_distributions)}")
print(f"First 10: {all_distributions[:10]}")

# Option 1: Use curated stable distributions (default - 16 distributions)
auto_fitter = processor.get_auto_fitter(criterion='rmse')
best_result = auto_fitter.fit_best_distribution()
print(f"Best: {best_result['distribution']} (RMSE: {best_result['rmse']:.4f})")

# Option 2: Use ALL 113 distributions (takes longer but comprehensive)
auto_fitter_all = processor.get_auto_fitter(
    candidates=all_distributions,  # All 113 distributions!
    criterion='rmse'
)
best_comprehensive = auto_fitter_all.fit_best_distribution()

# Option 3: Custom subset for your specific domain
wind_specific = ['weibull_min', 'lognorm', 'gamma', 'rayleigh', 'chi2']
auto_fitter_wind = processor.get_auto_fitter(
    candidates=wind_specific,
    criterion='rmse'
)

# Get comprehensive comparison
all_results = auto_fitter.get_comparison_table(sort_by='rmse')
for i, (dist, result) in enumerate(list(all_results.items())[:5]):
    if result['success']:
        print(f"{i+1}. {dist}: RMSE={result['rmse']:.4f}, AIC={result['aic']:.2f}")

# Use the best-fitted distribution directly
best_adjuster = auto_fitter.get_best_adjuster()
percentile_95 = best_adjuster.ppf(0.95)
print(f"95th percentile wind speed: {percentile_95:.2f} m/s")
```

### Available Distributions

**AutoFitter supports all 113 SciPy continuous distributions**, including:

- **Common**: weibull_min, lognorm, gamma, norm, expon, rayleigh, chi2, beta
- **Specialized**: gumbel_r, pareto, invgamma, maxwell, triang, laplace
- **Advanced**: genextreme, gengamma, levy_stable, johnsonsu, burr12
- **Complete list**: Use `AutoFitter.get_all_available_distributions()` to see all

**Default Strategy**: Uses a curated subset of 16 stable, commonly-used distributions to balance comprehensiveness with performance. Override with `candidates=AutoFitter.get_all_available_distributions()` for exhaustive testing.

### Selection Criteria Available

- **`rmse`**: Root Mean Square Error (lower is better) - **Default**
- **`aic`**: Akaike Information Criterion (lower is better)
- **`bic`**: Bayesian Information Criterion (lower is better)
- **`ks_pvalue`**: Kolmogorov-Smirnov p-value (higher is better)
- **`chi2_pvalue`**: Chi-square p-value (higher is better)

### Integration with Existing Architecture

`AutoFitter` seamlessly integrates with the existing factory pattern:

1. **DataProcessor** creates `AutoFitter` via `get_auto_fitter()`
2. **AutoFitter** creates multiple `MagicAdjuster` instances as needed
3. **MagicAdjuster** instances handle individual distribution fitting
4. **AutoFitter** compares results and selects the best fit

This three-tier architecture maintains the principle: **"Simple things should be simple, complex things should be possible"**. Basic operations remain straightforward, manual distribution fitting is available when needed, and automatic model selection is there for complex analysis!

## 🔄 The Method Interceptor: Making SciPy Methods Accessible

### What is a Method Interceptor?

A **method interceptor** is a programming technique that "catches" method calls that don't exist in your class and redirects them somewhere else. In MagicA, we use Python's special `__getattr__` method to achieve this magic.

### How It Works in MagicA

#### The Challenge
SciPy distributions have dozens of useful methods like `cdf()`, `pdf()`, `ppf()`, `rvs()`, etc. Without an interceptor, you'd need to:

1. **Manually wrap each method**:
```python
# Without interceptor (tedious approach)
def cdf(self, x):
    return self.fitted_distribution(*self.fitted_params).cdf(x)

def pdf(self, x):
    return self.fitted_distribution(*self.fitted_params).pdf(x)

def ppf(self, q):
    return self.fitted_distribution(*self.fitted_params).ppf(q)
# ... and 30+ more methods!
```

2. **Remember to update when SciPy adds new methods**
3. **Handle different parameter structures for each distribution**

#### Our Solution: Double-Layer Interceptor

We implement `__getattr__` in **both** classes for seamless method access:

```python
# In DataProcessor
def __getattr__(self, name):
    # First interceptor - delegates to MagicAdjuster
    if self._adjuster is None:
        raise AttributeError("Did you forget to call fit_distribution() first?")
    return getattr(self._adjuster, name)

# In MagicAdjuster  
def __getattr__(self, name):
    # Second interceptor - delegates to SciPy distribution
    frozen_dist = self.fitted_distribution(*self.fitted_params)
    if hasattr(frozen_dist, name):
        return getattr(frozen_dist, name)
    raise AttributeError(f"Method '{name}' not found")
```

### Step-by-Step Interceptor Flow

Let's trace what happens when you call `processor.cdf(5.0)`:

#### Step 1: Python Looks for `cdf` in DataProcessor
```python
processor = ma.read_data(wind_data)
processor.fit_distribution('weibull')
result = processor.cdf(5.0)  # ← Python starts here
```

**Python's internal search**:
1. ✅ Look in `processor.__dict__` → Not found
2. ✅ Look in `DataProcessor` class → Not found  
3. ✅ Look in parent classes → Not found
4. 🎯 **Call `DataProcessor.__getattr__('cdf')`**

#### Step 2: DataProcessor Interceptor Activates
```python
def __getattr__(self, name):  # name = 'cdf'
    if self._adjuster is None:
        raise AttributeError("Did you forget to call fit_distribution() first?")
    return getattr(self._adjuster, name)  # Delegate to MagicAdjuster
```

**What happens**:
- ✅ Check if `_adjuster` exists (it does, created during `fit_distribution`)
- 🔄 **Delegate to `getattr(self._adjuster, 'cdf')`**

#### Step 3: MagicAdjuster Interceptor Activates
```python
def __getattr__(self, name):  # name = 'cdf'  
    frozen_dist = self.fitted_distribution(*self.fitted_params)
    if hasattr(frozen_dist, name):
        return getattr(frozen_dist, name)  # Return SciPy method
    raise AttributeError(f"Method '{name}' not found")
```

**What happens**:
- 🏭 Create "frozen" distribution with fitted parameters
- ✅ Check if SciPy distribution has `cdf` method (it does!)
- 🎁 **Return the actual SciPy `cdf` method**

#### Step 4: Method Execution
```python
result = processor.cdf(5.0)  # Now calls the SciPy method directly
```

### Real-World Analogy: Restaurant Chain

Think of this like a restaurant chain with efficient delegation:

1. **You (Customer)**: "I want the chef's special pasta"
2. **Waiter (DataProcessor)**: "I don't cook, but let me ask the kitchen manager"
3. **Kitchen Manager (MagicAdjuster)**: "I don't cook either, but I'll get the specialist chef"
4. **Specialist Chef (SciPy Distribution)**: "Here's your perfect pasta!"

Each level knows exactly who to delegate to, and you get exactly what you asked for.

### Available Methods Through Interceptor

Once you fit a distribution, you automatically get access to **all** SciPy statistical methods:

#### Probability Functions with Smart Defaults
```python
processor.fit_distribution('weibull')

# Smart defaults - use original data when no input provided
pdf_values = processor.pdf()         # PDF at all original data points
cdf_values = processor.cdf()         # CDF at all original data points  
survival_values = processor.sf()     # Survival function at original data

# Custom inputs still work normally
prob = processor.cdf(10.0)           # P(X ≤ 10) at specific value
density = processor.pdf([5, 10, 15]) # PDF at custom points
survival = processor.sf(8.5)         # P(X > 8.5) at specific value
```

#### Why Smart Defaults Are Useful
The most common use case is evaluating the fitted distribution at your original data points:
```python
# Without smart defaults (tedious)
data = [1, 2, 3, 4, 5]
processor = ma.read_data(data)
processor.fit_distribution('weibull')
pdf_at_data = processor.pdf(data)    # Have to pass data again

# With smart defaults (elegant)
data = [1, 2, 3, 4, 5]  
processor = ma.read_data(data)
processor.fit_distribution('weibull')
pdf_at_data = processor.pdf()        # Automatically uses original data!
```

#### Other Statistical Methods
```python
# Percent Point Function (inverse CDF) - requires input
value = processor.ppf(0.95)          # Value where P(X ≤ value) = 0.95

# Random sampling - no input needed
samples = processor.rvs(size=1000)   # 1000 random samples
single_sample = processor.rvs()      # Single random value

# Statistical moments - no input needed
mean, var, skew, kurt = processor.stats(moments='mvsk')
print(f"Mean: {mean}, Variance: {var}")

# Entropy and median - no input needed
entropy = processor.entropy()
median = processor.median()
```

### Error Handling in Interceptors

The interceptors provide clear, helpful error messages:

#### Before Fitting Distribution
```python
processor = ma.read_data(data)
processor.cdf(5.0)
# Error: "DataProcessor object has no attribute 'cdf'. 
#         Did you forget to call fit_distribution() first?"
```

#### Invalid Method Name
```python
processor.fit_distribution('weibull')  
processor.nonexistent_method()
# Error: "MagicAdjuster object has no attribute 'nonexistent_method'"
```

### Performance Benefits

#### Memory Efficiency
- **No method duplication**: We don't store copies of SciPy methods
- **Lazy evaluation**: Methods are only accessed when called
- **Single source of truth**: SciPy remains the authoritative implementation

#### Maintenance Benefits  
- **Automatic updates**: New SciPy methods are immediately available
- **No version conflicts**: Always uses your installed SciPy version
- **Reduced code**: No need to maintain wrapper methods

### Advanced Example: Distribution Comparison

```python
import magica as ma
import numpy as np

# Generate test data
data = np.random.weibull(2, 1000) * 8 + 2

# Load and try different distributions
processor = ma.read_data(data)

distributions = ['weibull', 'gamma', 'lognorm']
comparison = {}

for dist_name in distributions:
    # Fit distribution  
    processor.fit_distribution(dist_name)
    
    # Use intercepted methods for analysis
    comparison[dist_name] = {
        'params': processor.get_fitted_params(),
        'mean': processor.stats(moments='m'),
        'median': processor.ppf(0.5),
        'p95': processor.ppf(0.95),
        'samples': processor.rvs(size=100)
    }

# All of this works seamlessly through the interceptor!
for dist, results in comparison.items():
    print(f"{dist}: mean={results['mean']:.2f}, p95={results['p95']:.2f}")
```

### Key Takeaways: Method Interceptor

1. **Transparency**: You interact with MagicA objects as if they were SciPy distributions
2. **Completeness**: Access to **all** SciPy statistical methods automatically  
3. **Error-friendly**: Clear messages guide you when something goes wrong
4. **Future-proof**: New SciPy features are immediately available
5. **Performance**: No overhead from method wrapping or duplication

The interceptor pattern makes MagicA feel like a natural extension of SciPy while providing the convenience and structure of a dedicated statistical analysis package!

## 🧠 Smart Wrapper: Automatic Data Usage

### The Problem with Traditional Approach

In standard SciPy usage, you constantly need to pass your data to evaluate distributions:

```python
# Traditional SciPy approach (repetitive)
import scipy.stats as stats
import numpy as np

data = [2.1, 3.4, 1.8, 4.2, 2.9, 3.1, 2.7]
params = stats.weibull_min.fit(data)
dist = stats.weibull_min(*params)

# Every time you want to evaluate at your data points:
pdf_values = dist.pdf(data)    # Have to pass data again
cdf_values = dist.cdf(data)    # Have to pass data again  
sf_values = dist.sf(data)      # Have to pass data again
```

### MagicA's Smart Solution

Our wrapper automatically uses your original data when no input is provided:

```python
# MagicA approach (elegant)
import magica as ma

data = [2.1, 3.4, 1.8, 4.2, 2.9, 3.1, 2.7]
processor = ma.read_data(data)
processor.fit_distribution('weibull')

# Automatic data usage - no need to pass data again!
pdf_values = processor.pdf()    # Uses original data automatically
cdf_values = processor.cdf()    # Uses original data automatically
sf_values = processor.sf()      # Uses original data automatically
```

### How the Smart Wrapper Works

The wrapper intelligently detects when you call certain methods without arguments:

```python
def smart_wrapper(*args, **kwargs):
    # If no positional arguments and method commonly uses data
    if len(args) == 0 and name in ['pdf', 'cdf', 'sf', 'logpdf', 'logcdf', 'logsf']:
        return original_method(self.data, **kwargs)  # Use original data
    else:
        return original_method(*args, **kwargs)      # Use provided arguments
```

### Smart Methods vs Regular Methods

| Method Type | Smart Default | Example Usage |
|-------------|---------------|---------------|
| **Smart Methods** | Use original data | `processor.pdf()` → PDF at data points |
| | | `processor.cdf()` → CDF at data points |
| | | `processor.sf()` → Survival at data points |
| **Regular Methods** | No default data | `processor.ppf(0.95)` → Must provide quantile |
| | | `processor.rvs(100)` → Must specify sample size |

### Flexibility: You Can Still Override

The smart wrapper doesn't limit you - you can always provide custom inputs:

```python
processor.fit_distribution('weibull')

# Use smart defaults
auto_pdf = processor.pdf()              # PDF at original data points

# Or provide custom inputs  
custom_pdf = processor.pdf([1, 2, 3])   # PDF at custom points
single_pdf = processor.pdf(2.5)         # PDF at single point
```

### Real-World Example: Wind Speed Analysis

```python
import magica as ma
import numpy as np

# Wind speed data (m/s)
wind_data = [3.2, 4.1, 2.8, 5.3, 3.9, 4.7, 2.1, 6.2, 3.5, 4.4]
processor = ma.read_data(wind_data)
processor.fit_distribution('weibull')

# Smart wrapper in action
print("=== Using Smart Defaults ===")
pdf_at_data = processor.pdf()           # PDF at all wind speed measurements
cdf_at_data = processor.cdf()           # CDF at all wind speed measurements
print(f"PDF values: {pdf_at_data}")
print(f"CDF values: {cdf_at_data}")

print("\n=== Using Custom Inputs ===")
pdf_at_5ms = processor.pdf(5.0)         # PDF at 5 m/s specifically
cdf_at_10ms = processor.cdf(10.0)       # CDF at 10 m/s specifically  
print(f"PDF at 5 m/s: {pdf_at_5ms}")
print(f"Probability of wind ≤ 10 m/s: {cdf_at_10ms}")

print("\n=== Methods that require input ===")
wind_95th = processor.ppf(0.95)         # 95th percentile wind speed
samples = processor.rvs(size=5)         # Generate 5 random wind speeds
print(f"95th percentile: {wind_95th:.2f} m/s")
print(f"Random samples: {samples}")
```

### Behind the Scenes: Method Detection

The smart wrapper categorizes methods into two groups:

#### Data-Aware Methods (use smart defaults):
- `pdf`, `cdf`, `sf` - Probability functions
- `logpdf`, `logcdf`, `logsf` - Log probability functions

#### Parameter-Required Methods (need explicit input):
- `ppf`, `isf` - Inverse functions (need quantiles)
- `rvs` - Random sampling (need sample size)
- `stats` - Statistical moments (no data input needed)
- `entropy`, `median` - Distribution properties (no data input needed)

### Error Prevention

The smart wrapper helps prevent common mistakes:

```python
processor.fit_distribution('weibull')

# This works - uses original data
result = processor.pdf()

# This also works - uses custom data
result = processor.pdf([1, 2, 3])

# This would fail clearly - ppf needs a quantile
try:
    result = processor.ppf()  # Missing required quantile argument
except TypeError as e:
    print("Error: ppf() missing required argument")
```

### Performance Considerations

**Efficiency**: The wrapper adds minimal overhead:
- Smart detection happens once per method call
- Original data is stored, not recalculated
- SciPy methods run at full speed

**Memory**: No additional memory usage:
- No data duplication
- Wrapper functions are lightweight
- Original SciPy distribution objects are reused

This smart wrapper makes MagicA incredibly intuitive for the most common use case: analyzing your fitted distribution at your original data points!
