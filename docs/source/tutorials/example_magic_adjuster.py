# Example of using MagicA's new fluent interface
import magica as ma
import numpy as np

# Example data (simulating wind speeds)
wind_data = [2.1, 5.4, 8.7, 12.3, 6.8, 9.1, 15.2, 3.4, 7.6, 11.0, 4.5, 13.2, 8.9, 6.7, 10.5]

print("=== MagicA - Fluent Interface ===")
print(f"Original data: {wind_data[:5]}... (total: {len(wind_data)} points)")

# 1. Read data with the new interface
data = ma.read_data(wind_data)
print(f"\n1. Data loaded: {data}")

# 2. Get basic statistics
stats = data.get_basic_stats()
print(f"\n2. Basic statistics:")
print(f"   Mean: {stats['mean']:.2f}")
print(f"   Standard deviation: {stats['std']:.2f}")
print(f"   Minimum: {stats['min']:.2f}")
print(f"   Maximum: {stats['max']:.2f}")

# 3. Fit Weibull distribution
fitted_data = data.fit_distribution('weibull')
print(f"\n3. Distribution fitted: {fitted_data}")

# 4. Get fitted parameters
params = fitted_data.get_fitted_params()
print(f"\n4. Weibull parameters: {params}")

# 5. Get complete distribution information
info = fitted_data.get_distribution_info()
print(f"\n5. Distribution information:")
for key, value in info.items():
    print(f"   {key}: {value}")

# 6. Test other distributions in sequence
print(f"\n6. Testing other distributions:")

# Gamma
data.fit_distribution('gamma')
gamma_params = data.get_fitted_params()
print(f"   Gamma: {gamma_params}")

# Normal
data.fit_distribution('norm')
norm_params = data.get_fitted_params()
print(f"   Normal: {norm_params}")

# Lognormal
data.fit_distribution('lognorm')
lognorm_params = data.get_fitted_params()
print(f"   Lognormal: {lognorm_params}")

# 7. Demonstrate fluent interface (method chaining)
print(f"\n7. Fluent interface (method chaining):")
result = ma.read_data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).fit_distribution('weibull')
print(f"   Result: {result}")
print(f"   Parameters: {result.get_fitted_params()}")

print("\n=== Example completed successfully! ===")
