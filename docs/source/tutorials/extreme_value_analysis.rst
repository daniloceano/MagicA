Extreme Value Analysis
======================

Learn how to extract extreme observations and analyze return periods and return
values. These tutorials use synthetic wind data and include their figures and
outputs.

Start with :doc:`/quickstart` and basic familiarity with pandas time series.
A datetime index is important for block extraction, time-window declustering,
and interpreting the observation period.

Recommended Order
-----------------

**1. Extreme Value Analysis Tutorial** introduces block maxima, peaks over
threshold, declustering, return levels, and diagnostic plots. It includes a
five-day pulse example showing the dates selected by each declustering method.

**2. Directional Extreme Wind Analysis Tutorial** extends the workflow to
wind-direction sectors, threshold searches, result tables, and polar plots.
Read the first tutorial before proceeding to this directional example.

.. toctree::
   :maxdepth: 1

   extremes_tutorial
   directional_extremes_tutorial

Related Guides
--------------

- :doc:`/api/extremes` documents ExtremesAnalyzer, EVAFit, and the distinction
  between explicit EVA fitting and the legacy fitting interface.
- :doc:`/api/utils` documents the synthetic wind-data generators.
- :doc:`distribution_fitting` covers fitting distributions to general samples.
