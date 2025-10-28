Utility Functions
=================

The utils module provides utility functions for data generation, preprocessing, and analysis support.

Synthetic Data Generation
--------------------------

MagicA includes synthetic data generators for creating realistic test datasets. These are particularly useful for:

- **Testing and validation**: Verify extreme value analysis methods
- **Tutorial demonstrations**: Clear, reproducible examples
- **Method comparison**: Benchmark different approaches
- **Understanding behavior**: Explore how parameters affect results

Module Reference
----------------

.. automodule:: magica.utils.synthetic_data
   :members:
   :undoc-members:
   :show-inheritance:

Quick Start
-----------

Generate Non-Directional Wind Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create synthetic wind speed time series with storms and seasonal variations:

.. code-block:: python

    from magica.utils import generate_wind_data
    import matplotlib.pyplot as plt
    
    # Generate 30 years of daily wind data
    wind_series, plots = generate_wind_data(
        n_years=30,
        freq='D',
        mean_wind=8.0,
        weibull_shape=2.5,
        seasonal_amplitude=0.3,
        n_storms_per_year=5,
        storm_duration_days=(2, 5),
        storm_intensity_range=(12, 20),
        random_seed=42,
        create_plots=True
    )
    
    # Display plots
    plt.show()
    
    # Use the data
    print(f"Generated {len(wind_series)} observations")
    print(f"Max wind speed: {wind_series.max():.2f} m/s")

Without Plots
^^^^^^^^^^^^^

For production use, omit the plots:

.. code-block:: python

    # Just get the data (no plots)
    wind_series = generate_wind_data(
        n_years=30,
        freq='D',
        random_seed=42,
        create_plots=False  # Default
    )

Generate Directional Wind Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create synthetic wind data with directional characteristics:

.. code-block:: python

    from magica.utils import generate_directional_wind_data
    
    # Generate 10 years of hourly directional wind data
    wind_data, plots = generate_directional_wind_data(
        n_years=10,
        freq='H',
        mean_wind=8.0,
        weibull_shape=2.0,
        seasonal_amplitude=0.25,
        n_storms_per_year=5,
        storm_duration_hours=(8, 16),
        storm_intensity_range=(15, 25),
        prevailing_direction=270,  # West
        directional_concentration=1.5,
        directional_speed_factors={
            'W': 1.4,   # Higher speeds from West (fetch effect)
            'SW': 1.4,  # Higher speeds from Southwest
            'N': 0.8,   # Lower speeds from North (land effect)
            'NE': 0.8   # Lower speeds from Northeast
        },
        storm_directions=[250, 270, 290],  # Storms from W/SW
        random_seed=42,
        create_plots=True
    )
    
    plt.show()
    
    # Access the data
    print(wind_data.head())
    print(f"Columns: {wind_data.columns.tolist()}")

Customizing Parameters
~~~~~~~~~~~~~~~~~~~~~~

Both generators support extensive customization:

**Base Distribution:**

.. code-block:: python

    wind_series = generate_wind_data(
        mean_wind=10.0,       # Mean wind speed (m/s)
        weibull_shape=2.5,    # Shape parameter (typical: 1.5-3.0)
        seasonal_amplitude=0.4  # Seasonal variation (0.3 = ±30%)
    )

**Storm Characteristics:**

.. code-block:: python

    wind_series = generate_wind_data(
        n_storms_per_year=8,              # More frequent storms
        storm_duration_days=(1, 3),       # Shorter storms
        storm_intensity_range=(10, 15),   # Less intense
        storm_decay_shape='gaussian'      # Gaussian vs triangular
    )

**Directional Features:**

.. code-block:: python

    wind_data = generate_directional_wind_data(
        prevailing_direction=180,         # South
        directional_concentration=2.0,    # Stronger directional bias
        secondary_direction=90,           # East
        secondary_concentration=1.0,
        secondary_fraction=0.3,          # 30% from secondary direction
        directional_speed_factors={       # Speed modulation by direction
            'N': 0.7,
            'S': 1.3,
            'E': 0.9,
            'W': 1.1
        }
    )

Custom Time Range
~~~~~~~~~~~~~~~~~

Specify exact date ranges:

.. code-block:: python

    import pandas as pd
    
    # Use custom dates
    wind_series = generate_wind_data(
        n_years=None,  # Ignore
        freq='6H',     # 6-hourly data
        start_date='2000-01-01',
        end_date='2020-12-31'
    )

Integration with MagicA
~~~~~~~~~~~~~~~~~~~~~~~

The synthetic data integrates seamlessly with MagicA's analysis tools:

.. code-block:: python

    import magica as ma
    from magica.utils import generate_wind_data
    
    # Generate data
    wind_series = generate_wind_data(n_years=30, freq='D')
    
    # Analyze extremes
    processor = ma.read_data(wind_series)
    extremes = processor.get_extremes_analyzer(time_unit='years')
    
    # Extract annual maxima
    annual_maxima, years = extremes.extract_block_maxima(block_size='Y')
    
    # Fit GEV distribution
    extremes_bm = ma.read_data(annual_maxima).get_extremes_analyzer()
    extremes_bm.fit_distribution('genextreme')
    
    # Calculate return values
    return_periods = [10, 20, 50, 100]
    for rp in return_periods:
        rv = extremes_bm.calculate_return_value(rp)
        print(f"{rp}-year return value: {rv:.2f} m/s")

Use Cases
---------

Testing Extreme Value Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate your extreme value analysis workflow:

.. code-block:: python

    from magica.utils import generate_wind_data
    import magica as ma
    
    # Generate data with known characteristics
    wind_series = generate_wind_data(
        n_years=50,
        n_storms_per_year=5,
        storm_intensity_range=(15, 25),
        random_seed=123
    )
    
    # Apply POT analysis
    processor = ma.read_data(wind_series)
    extremes = processor.get_extremes_analyzer(time_unit='years')
    
    # Use intelligent threshold selection
    result = extremes.find_optimal_pot_threshold(
        min_samples=50,
        percentile_min=90,
        percentile_max=99,
        verbose=True
    )
    
    print(f"Optimal threshold: {result['threshold']:.2f} m/s")
    print(f"Independent exceedances: {result['n_independent']}")

Directional Extreme Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate data for directional wind studies:

.. code-block:: python

    from magica.utils import generate_directional_wind_data
    import magica as ma
    
    # Generate directional data
    wind_data = generate_directional_wind_data(
        n_years=10,
        freq='H',
        prevailing_direction=270,  # West
        n_storms_per_year=5
    )
    
    # Analyze by sector
    sectors = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    
    for sector in sectors:
        # Filter by direction
        sector_mask = wind_data['sector_name'] == sector
        sector_series = wind_data.loc[sector_mask, 'wind_speed']
        
        # Analyze extremes
        processor = ma.read_data(sector_series)
        extremes = processor.get_extremes_analyzer(time_unit='years')
        
        # ... continue analysis

See Also
--------

- :doc:`extremes` - Extreme value analysis tools
- :doc:`auto_fitter` - Automatic distribution fitting
- :doc:`../tutorials/extremes_tutorial` - Extreme value analysis tutorial
- :doc:`../tutorials/directional_extremes_tutorial` - Directional extremes tutorial

Function Reference
------------------

generate_wind_data
~~~~~~~~~~~~~~~~~~

.. autofunction:: magica.utils.synthetic_data.generate_wind_data

generate_directional_wind_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: magica.utils.synthetic_data.generate_directional_wind_data
