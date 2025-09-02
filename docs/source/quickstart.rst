Quick Start
===========

MagicA (Magic Adjustment) is a Python package for statistical data adjustment with a focus on distribution fitting and Monte Carlo stability analysis.

Installation
------------

Install MagicA using pip:

.. code-block:: bash

    pip install magica

Basic Usage
-----------

Distribution Fitting
~~~~~~~~~~~~~~~~~~~~

Start by fitting a distribution to your data:

.. code-block:: python

    import numpy as np
    from magica.core import MagicAdjuster
    
    # Generate sample data (Weibull distribution)
    data = np.random.weibull(2, 1000)
    
    # Create adjuster and fit distribution
    adjuster = MagicAdjuster(data)
    adjuster.fit_distribution('weibull_min')
    
    # Get fitted parameters
    params = adjuster.get_fitted_params()
    print(f"Fitted parameters: {params}")

Goodness-of-Fit Testing
~~~~~~~~~~~~~~~~~~~~~~~

Evaluate how well your distribution fits the data:

.. code-block:: python

    # Perform chi-square test
    chi2_result = adjuster.goodness_of_fit('chi2')
    print(f"Chi-square p-value: {chi2_result['p_value']}")
    
    # Perform Kolmogorov-Smirnov test
    ks_result = adjuster.goodness_of_fit('ks')
    print(f"KS p-value: {ks_result['p_value']}")
    
    # Calculate RMSE
    rmse_result = adjuster.goodness_of_fit('rmse')
    print(f"RMSE: {rmse_result['rmse']}")

Monte Carlo Stability Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determine the minimum sample size needed for stable parameter estimation:

.. code-block:: python

    # Basic Monte Carlo analysis
    results = adjuster.monte_carlo_fit(
        n_repeats=100,
        tests=['chi2', 'ks']  # default: no summary figure
    )
    
    # Access results with xarray
    ks_pvalues = results['ks_pvalue']
    param_medians = results['param_0'].median(dim='repeats')
    
    # Check stability points
    stability = results.attrs['stability_points']
    print(f"Recommended minimum size: {stability['param_0']['size']}")

Working with xarray Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Monte Carlo analysis returns an xarray Dataset for easy data manipulation:

.. code-block:: python

    # Select data for specific sample size
    size_200_results = results.sel(sizes=200)
    
    # Calculate statistics across repeats
    param_std = results['param_0'].std(dim='repeats')
    ks_median = results['ks_pvalue'].median(dim='repeats')
    
    # Plot results directly
    results['ks_pvalue'].plot(x='sizes')
    
    # Convert to pandas for further analysis
    df = results.to_dataframe()

Advanced Examples
-----------------

Custom Binning Strategy
~~~~~~~~~~~~~~~~~~~~~~~

Control the binning strategy for chi-square tests:

.. code-block:: python

    # Use Scott's rule for binning
    results = adjuster.monte_carlo_fit(
        n_repeats=200,
        tests=['chi2'],
        bins='scott',
        plot_type='boxplots',
        fig_output_path='chi2_boxplots.png'
    )
    
    # Access chi-square test results
    chi2_pvalues = results['chi2_pvalue']

Pre-calculated Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

Use known distribution parameters instead of fitting (special use case):

.. code-block:: python

    # Use known Weibull parameters (shape=2, loc=0, scale=1)
    known_params = (2.0, 0.0, 1.0)
    
    results = adjuster.monte_carlo_fit(
        distribution_params=known_params,
        n_repeats=150,
        tests=['chi2', 'ks', 'rmse']
    )

Custom Fitting Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply parameter constraints during fitting:

.. code-block:: python

    # Fix location parameter for Weibull distribution
    results = adjuster.monte_carlo_fit(
        n_repeats=100,
        tests=['chi2', 'ks'],
        fit_kwargs={'floc': 0}  # Force location = 0
    )
    
    # Multiple constraints
    results = adjuster.monte_carlo_fit(
        n_repeats=100,
        tests=['chi2', 'ks'],
        fit_kwargs={'floc': 0, 'method': 'MLE'}
    )

Pre-calculated Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

Use known distribution parameters instead of fitting:

.. code-block:: python

    # Use known Weibull parameters (shape=2, loc=0, scale=1)
    known_params = (2.0, 0.0, 1.0)
    
    results = adjuster.monte_carlo_fit(
        distribution_params=known_params,
        n_repeats=150,
        tests=['chi2', 'ks', 'rmse']
    )
    
    # Parameters are consistent across all repeats
    assert np.allclose(results['param_0'].values, 2.0)

Custom Fitting Options
~~~~~~~~~~~~~~~~~~~~~~

Pass additional arguments to the fitting process:

.. code-block:: python

    results = adjuster.monte_carlo_fit(
        n_repeats=100,
        tests=['chi2', 'ks'],
        fit_kwargs={'method': 'MLE', 'optimizer': 'powell'}
    )
    
    # Visualize parameter convergence
    results['param_0'].plot(x='sizes', hue='repeats', alpha=0.3)

Understanding Results
---------------------

The `monte_carlo_fit` method returns an xarray Dataset with:

**Dimensions:**
- **sizes**: Sample sizes tested (e.g., [50, 100, 150, 200, ...])
- **repeats**: Repetition index for each size (e.g., [0, 1, 2, ..., 19])

**Data Variables:**
- **param_0, param_1, ...**: Fitted distribution parameters for each size/repeat
- **ks_statistic, ks_pvalue**: Kolmogorov-Smirnov test results
- **chi2_statistic, chi2_pvalue**: Chi-square test results
- **rmse**: Root mean square error values

**Attributes:**
- **stability_points**: Detected stability points for each variable
- **figure_path**: Path to saved summary figure (only if generated)

**Easy Data Access:**

.. code-block:: python

    # Get all KS p-values
    ks_data = results['ks_pvalue']
    
    # Select specific size
    size_100_data = results.sel(sizes=100)
    
    # Calculate median across repeats
    param_medians = results['param_0'].median(dim='repeats')
    
    # Check when parameters stabilize
    stability = results.attrs['stability_points']['param_0']
    print(f"Parameter 0 stabilizes at size: {stability['size']}")

Generating a Summary Figure
---------------------------

By default no figure is created (faster). Provide a `fig_output_path` to save a
2x3 summary figure (first row: up to 3 parameters; second row: test p-values / RMSE):

.. code-block:: python

    # Generate and save figure with series style panels
    results = adjuster.monte_carlo_fit(
        tests=['ks','chi2','rmse'],
        fig_output_path='stability_summary.png'
    )

    # Boxplot style
    results = adjuster.monte_carlo_fit(
        tests=['ks','chi2'],
        plot_type='boxplots',
        fig_output_path='stability_boxplots.png'
    )

    # Path stored in attributes
    print(results.attrs.get('figure_path'))  # saved file path

The red dashed vertical line in each panel marks the first sample size where
the corresponding parameter or test metric meets the stability criterion.

You can still craft custom visualizations directly with xarray / matplotlib:

.. code-block:: python

    results['param_0'].plot(x='sizes')

Supported Distributions
-----------------------

MagicA supports all continuous distributions from scipy.stats, including:

- `'norm'` - Normal distribution
- `'weibull_min'` - Weibull distribution
- `'gamma'` - Gamma distribution
- `'exponweib'` - Exponentiated Weibull
- `'lognorm'` - Log-normal distribution
- And many more...

Next Steps
----------

- Explore the :doc:`tutorials/index` for detailed examples
- Check the :doc:`api/core` for complete API documentation
- See :doc:`contributing` for development guidelines
