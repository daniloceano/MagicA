Tutorials
=========

Choose a tutorial according to the task you want to complete. Each notebook
includes executable examples, figures, and saved outputs.

.. tip::
   **New to MagicA?** Begin with :doc:`/installation` and :doc:`/quickstart`.

.. _which-tutorial-should-i-use:

Interactive Jupyter Notebooks
-----------------------------

.. _magicadjuster-tutorial:
.. _autofitter-tutorial:

**Distribution fitting:** fit a chosen distribution with MagicAdjuster, or
compare candidate distributions with AutoFitter.

.. _extreme-value-analysis-tutorial:

**Extreme value analysis:** start with block maxima and peaks over threshold,
then continue to the directional wind example.

.. toctree::
   :maxdepth: 2

   distribution_fitting
   extreme_value_analysis

API Usage Tutorial
------------------

.. _magica-api-usage:
.. _basic-example-weibull-fit-and-goodness-of-fit-evaluation:
.. _monte-carlo-stability-analysis:
.. _advanced-monte-carlo-usage:
.. _api-reference:
.. _read-data-data:
.. _dataprocessor-fit-distribution-kwargs:
.. _fitresult-params-and-fitresult-info:
.. _fitresult-goodness-of-fit-method-kwargs:
.. _magicadjuster-monte-carlo-fit:
.. _dataprocessor-get-basic-stats:
.. _working-with-xarray-results:
.. _goodness-of-fit-methods:

For common operations and code patterns, see :doc:`api_usage`.
The guide is available on its own page rather than reproduced in this index.

Monte Carlo Stability Tutorial
------------------------------

.. _why-monte-carlo-stability:
.. _methodology-what-we-do:
.. _inputs-and-options:
.. _stability-detection-methods:
.. _cv-method-coefficient-of-variation:
.. _kneedle-method-elbow-detection:
.. _plateau-method-relative-gain-heuristic:
.. _method-selection-guide:
.. _sampling-strategies:
.. _reproducibility-and-seed:
.. _practical-guidance:
.. _outputs:
.. _quick-example:
.. _interpretation-tips:
.. _method-comparison:
.. _further-reading:
.. _scientific-context:

For the motivation, sampling strategies, and stability-detection options,
see :doc:`monte_carlo`. The guide distinguishes the scientific context from
MagicA's implementation options.

.. toctree::
   :hidden:

   api_usage
   monte_carlo

Legacy Examples
---------------

.. _weibull-fit-example:
.. _magic-adjuster-example:

The historical files ``MagicA_Weibull_Fit_Example.ipynb`` and
``example_magic_adjuster.py`` are not included in the current repository.
Use :doc:`magic_adjuster_tutorial` for an executable example.

Downloadable Files
------------------

Download the notebooks to run the tutorials locally:

- :download:`MagicAdjuster tutorial <magic_adjuster_tutorial.ipynb>`
- :download:`AutoFitter tutorial <auto_fitter_tutorial.ipynb>`
- :download:`Extreme Value Analysis tutorial <extremes_tutorial.ipynb>`
- :download:`Directional Extremes tutorial <directional_extremes_tutorial.ipynb>`
- :download:`API Usage guide (Markdown) <api_usage.md>`

.. _best-practices-summary:

Next Steps
----------

Consult :doc:`/api/core`, :doc:`/api/auto_fitter`, and :doc:`/api/extremes`
for method signatures and return types. For Monte Carlo options, see
:doc:`/api/monte_carlo`. Contribution information is in :doc:`/contributing`.
