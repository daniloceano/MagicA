MagicA Documentation
====================

.. image:: _static/magica_logo_whitebg.svg
   :alt: MagicA Logo
   :align: center
   :width: 300px

|

**MagicA** (Magic Adjustment) is a Python package for statistical data adjustment, with special focus on wind data, including advanced fitting techniques, goodness-of-fit tests, and visualization.

Overview
--------

MagicA provides tools for:

* Statistical distribution fitting (Weibull, Normal, Lognormal, etc.)
* Goodness-of-fit tests (Kolmogorov-Smirnov, Chi-square, etc.)
* Automatic best distribution selection
* Integrated visualization functions
* Specialized wind data analysis
* Advanced statistical fitting techniques

Getting Started
---------------

.. _installation:

Start with :doc:`installation` to set up MagicA.

.. _quick-start:

Follow :doc:`quickstart` for your first fit, or explore :doc:`tutorials/index`
for guided examples with figures.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Methodology

   Monte Carlo Stability <tutorials/monte_carlo>

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/auto_fitter
   api/extremes
   api/monte_carlo
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
