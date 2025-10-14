Tutorials
=========

This section contains comprehensive tutorials and examples for using MagicA for statistical distribution fitting and Monte Carlo stability analysis.

.. tip::
   **New to MagicA?** Start with the :doc:`/quickstart` guide, then explore the interactive Jupyter notebooks below.

Interactive Jupyter Notebooks
-----------------------------

MagicAdjuster Tutorial
~~~~~~~~~~~~~~~~~~~~~~

**Comprehensive guide to single distribution fitting with Monte Carlo CPS (Coefficient/P-value/Sample size) methods.**

This tutorial covers:

- Distribution fitting and parameter estimation
- Goodness-of-fit testing (KS, Chi-square, RMSE)
- Different binning strategies for histogram-based tests
- **Monte Carlo stability analysis** for determining minimum sample sizes
- **Large Sample Size Effect** - why p-values become unreliable with large datasets (>10,000 samples)
- **RMSE as the most reliable stability indicator**
- Comparing different sampling strategies (random, bootstrap, disjoint)
- CPS (Coefficient/P-value/Sample size) method from Lin et al. (2011)
- Real-world example with INMET weather station data

.. important::
   This tutorial demonstrates why **RMSE is the preferred metric** for assessing fit quality and stability, especially with large datasets where p-values can be misleading.

.. toctree::
   :maxdepth: 1

   magic_adjuster_tutorial

**Key Learning Outcomes:**

- Master Monte Carlo stability analysis
- Understand when p-values are reliable vs. unreliable
- Learn to use RMSE as primary fit quality indicator
- Determine optimal sample sizes for your data

AutoFitter Tutorial
~~~~~~~~~~~~~~~~~~~

**Learn how to automatically find the best distribution from 113+ scipy.stats options.**

This tutorial covers:

- Automatic distribution testing and selection
- **RMSE-based selection** (recommended for all data types)
- Testing default curated list (16 distributions) vs. comprehensive set (113+)
- Creating domain-specific distribution lists (e.g., wind, rainfall)
- Comparison tables and ranking by different criteria
- **Filtering by p-value for synthetic data** (with warnings for real-world use)
- Understanding when each selection criterion is appropriate
- Using the best-fitted distribution for further analysis

.. note::
   This tutorial emphasizes **RMSE as the primary criterion** for distribution selection, with clear guidance on when p-value filtering is appropriate vs. problematic.

.. toctree::
   :maxdepth: 1

   auto_fitter_tutorial

**Key Learning Outcomes:**

- Automate distribution selection with confidence
- Choose appropriate selection criteria for your data
- Understand trade-offs between speed (default list) and thoroughness (comprehensive)
- Apply best practices for real-world vs. synthetic data

API Usage Tutorial
------------------

Quick reference guide for common MagicA operations and API patterns.

.. include:: api_usage.md
   :parser: myst_parser.sphinx_

Monte Carlo Stability Tutorial
------------------------------

Detailed explanation of Monte Carlo methods for stability analysis and sample size determination.

.. include:: monte_carlo.rst
   :parser: restructuredtext

Which Tutorial Should I Use?
-----------------------------

**Choose MagicAdjuster Tutorial if:**

- ✓ You know which distribution to fit (e.g., Weibull for wind speed)
- ✓ You want to perform detailed stability analysis
- ✓ You need to understand parameter uncertainty
- ✓ You're working with large datasets (>10,000 samples)
- ✓ You want to learn about the large sample size effect

**Choose AutoFitter Tutorial if:**

- ✓ You're unsure which distribution fits your data best
- ✓ You want to compare multiple distributions quickly
- ✓ You need a comprehensive distribution search
- ✓ You're exploring a new dataset
- ✓ You want to validate your distribution choice

**Both tutorials emphasize:**

- ⭐ **RMSE as the most reliable metric** for fit quality
- 📊 When p-values are trustworthy vs. misleading
- 🎯 Best practices for real-world data analysis

Legacy Examples
---------------

Weibull Fit Example
~~~~~~~~~~~~~~~~~~~

.. note::
   The Jupyter notebook `MagicA_Weibull_Fit_Example.ipynb` can be found in this directory.
   You can download and run it locally to follow along with the examples.
   
   **Note:** This is a legacy example. We recommend using the updated MagicAdjuster tutorial instead.

Magic Adjuster Example
~~~~~~~~~~~~~~~~~~~~~~

.. note::
   The Python script `example_magic_adjuster.py` contains example usage of the MagicAdjuster class.
   You can find it in this directory and run it as a standalone script.
   
   **Note:** This is a legacy example. We recommend using the updated tutorials instead.

Downloadable Files
------------------

Download the following files to run locally:

**Main Tutorials:**

* :download:`magic_adjuster_tutorial.ipynb <magic_adjuster_tutorial.ipynb>` - Complete MagicAdjuster tutorial with Monte Carlo CPS methods
* :download:`auto_fitter_tutorial.ipynb <auto_fitter_tutorial.ipynb>` - Comprehensive AutoFitter tutorial with RMSE emphasis

**Legacy Materials:**

* :download:`MagicA_Weibull_Fit_Example.ipynb <MagicA_Weibull_Fit_Example.ipynb>` - Legacy Weibull fitting examples
* :download:`example_magic_adjuster.py <example_magic_adjuster.py>` - Python script with MagicAdjuster examples
* :download:`api_usage.md <api_usage.md>` - API usage guide in Markdown format

Best Practices Summary
----------------------

Based on the tutorials, here are the key best practices:

**Distribution Selection:**

1. **Use RMSE as primary criterion** for real-world data
2. Start with AutoFitter's default list for quick screening
3. Use comprehensive search when you need the absolute best fit
4. Create domain-specific lists for faster, targeted analysis

**Monte Carlo Analysis:**

1. **Always include 'rmse' in tests** - most reliable stability indicator
2. Use 100+ repeats for robust stability detection
3. Generate summary figures with `fig_output_path` for visual inspection
4. Check RMSE stability first, then validate with other metrics

**Large Sample Size Effect:**

1. **Avoid p-value-only selection** with large datasets (>10,000 samples)
2. Use RMSE for datasets of any size - it's always reliable
3. Understand that p-values can reject even excellent fits with large N
4. See MagicAdjuster tutorial for detailed examples

Next Steps
----------

After completing the tutorials:

- Apply MagicA to your own data
- Explore the :doc:`/api/core` and :doc:`/api/auto_fitter` documentation
- Read about :doc:`/api/monte_carlo` for advanced stability analysis
- Check :doc:`/contributing` to contribute to the project