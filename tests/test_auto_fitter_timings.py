import numpy as np
import time
import magica as ma
from magica.core.auto_fitter import AutoFitter


def test_auto_fitter_timings():
    """
    Test timing for fitting all available distributions on synthetic wind data.
    Prints timing, RMSE, and success for each distribution.
    """
    # Synthetic data (same as tutorial)
    np.random.seed(123)
    low_wind = np.random.weibull(2, 400) * 6 + 1
    normal_wind = np.random.lognormal(2, 0.5, 400)
    high_wind = np.random.gamma(3, 2, 200)
    wind_data = np.concatenate([low_wind, normal_wind, high_wind])

    # Get all available distributions
    all_distributions = AutoFitter.get_all_available_distributions()

    timings = []
    results = []

    print(f"{'Distribution':<20} {'Time (s)':<10} {'RMSE':<12} {'Success'}")
    print("-" * 50)
    for dist_name in all_distributions:
        processor = ma.read_data(wind_data)
        auto_fitter = processor.get_auto_fitter(candidates=[dist_name], criterion='rmse')
        start = time.time()
        try:
            result = auto_fitter.fit_best_distribution()
            elapsed = time.time() - start
            rmse = result.get('rmse', None)
            success = result.get('success', False)
        except Exception as e:
            elapsed = time.time() - start
            rmse = None
            success = False
        timings.append((dist_name, elapsed))
        results.append((dist_name, rmse, success))
        rmse_str = f"{rmse:.6f}" if rmse is not None else "-"
        print(f"{dist_name:<20} {elapsed:<10.4f} {rmse_str:<12} {success}")

    print("\nSummary Table:")
    print(f"{'Distribution':<20} {'Time (s)':<10} {'RMSE':<12} {'Success'}")
    print("-" * 50)
    for dist_name, elapsed in timings:
        rmse = next(r for r in results if r[0] == dist_name)[1]
        success = next(r for r in results if r[0] == dist_name)[2]
        rmse_str = f"{rmse:.6f}" if rmse is not None else "-"
        print(f"{dist_name:<20} {elapsed:<10.4f} {rmse_str:<12} {success}")

    total_time = sum(t for _, t in timings)
    print(f"\nTotal time for all distributions: {total_time:.2f} seconds")

if __name__ == "__main__":
    test_auto_fitter_timings()
