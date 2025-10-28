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
* Goodness-of-fit tests (Kolmogorov-Smirnov, Anderson-Darling, etc.)  
* Automatic best distribution selection
* Integrated visualization functions
* Specialized wind data analysis
* Advanced statistical fitting techniques

Installation
------------

For development installation:

.. code-block:: bash

   git clone https://github.com/daniloceano/MagicA.git
   cd MagicA
   pip install -e .

Quick Start
-----------

.. code-block:: python

   import magica as ma

   # Load data
   processor = ma.DataProcessor()
   processor.load_data('wind_data.csv')

   # Get basic statistics
   stats = processor.get_basic_stats()
   print(stats)

   # Get data as numpy array
   data_array = processor.get_data_array()

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
