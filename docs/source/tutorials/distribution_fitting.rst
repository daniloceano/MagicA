Distribution Fitting
====================

Learn how to fit probability distributions and inspect the results. Start with
:doc:`/quickstart` if you have not yet loaded data into MagicA.

Choose Your Starting Point
--------------------------

**You know which distribution to fit:** follow the MagicAdjuster tutorial.
It introduces fitting, the returned ``FitResult``, goodness-of-fit measures,
and plots. The later sections explore constraints and Monte Carlo stability.

**You want to compare distributions:** follow the AutoFitter tutorial.
It introduces candidate lists, comparison tables, selection criteria, and
using the selected ``FitResult`` for further analysis.

Both tutorials use synthetic data and include their figures and outputs.
Basic familiarity with NumPy arrays and Python is sufficient to get started.

.. toctree::
   :maxdepth: 1

   magic_adjuster_tutorial
   auto_fitter_tutorial

Related Guides
--------------

- :doc:`monte_carlo` explains subsampling and stability detection.
- :doc:`api_usage` provides common usage patterns.
- :doc:`/api/core` documents DataProcessor, MagicAdjuster, and FitResult.
- :doc:`/api/auto_fitter` documents automatic distribution selection.
- :doc:`extreme_value_analysis` introduces the separate workflow for extremes.
