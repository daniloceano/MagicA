Installation
============

MagicA (Magic Adjustment) can be installed using pip or from source for development.

Requirements
------------

MagicA requires Python 3.8 or higher and depends on:

* **NumPy** (>=1.20.0) - Numerical computing
* **SciPy** (>=1.7.0) - Scientific computing and statistical distributions
* **Pandas** (>=1.3.0) - Data structures and analysis
* **Matplotlib** (>=3.4.0) - Plotting and visualization
* **xarray** (>=0.16.0) - Labeled multi-dimensional arrays
* **Seaborn** (>=0.11.0) - Visualization
* **scikit-learn** (>=1.0.0) - Statistical tools

Install via pip
---------------

Install this MagicA project directly from its repository:

.. code-block:: bash

    pip install "git+https://github.com/daniloceano/MagicA.git"

This installs the dependencies declared by this repository. The PyPI project
named ``magica`` currently belongs to a different project; do not use the bare
``pip install magica`` command for this repository.

Install from Source
-------------------

For development or to get the latest features:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/daniloceano/MagicA.git
    cd MagicA
    
    # Install in development mode
    pip install -e .

Development Installation
------------------------

For contributing to MagicA, install with development dependencies:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/daniloceano/MagicA.git
    cd MagicA
    
    # Install with development dependencies
    pip install -e ".[dev]"

This includes additional packages for:

* **pytest** - Testing framework
* **jupyter** - Notebook examples
* **black** - Code formatting
* **flake8** - Code linting

Verify Installation
-------------------

To verify that MagicA is correctly installed:

.. code-block:: python

    import magica as ma
    import numpy as np
    
    # Generate sample data
    data = np.random.weibull(2, 1000)
    
    # Create processor and fit distribution
    processor = ma.read_data(data)
    fit = processor.fit('weibull_min')
    
    # Get fitted parameters
    params = fit.params
    print(f"✓ MagicA installed successfully!")
    print(f"Fitted parameters: {params}")

Optional Dependencies
---------------------

For Jupyter notebook support:

.. code-block:: bash

    pip install jupyter notebook

For documentation builds, also install the documentation requirements:

.. code-block:: bash

    pip install -r docs/requirements.txt

``tqdm`` is optional for progress bars and is included in the development extras.
Seaborn is already a required dependency.

Upgrading
---------

To upgrade to the latest version:

.. code-block:: bash

    pip install --upgrade "git+https://github.com/daniloceano/MagicA.git"

Uninstalling
------------

To remove MagicA:

.. code-block:: bash

    pip uninstall magica

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'magica'**

Make sure MagicA is installed in the active Python environment:

.. code-block:: bash

    pip show magica

**SciPy version conflicts**

Ensure you have a compatible SciPy version:

.. code-block:: bash

    pip install --upgrade "scipy>=1.7.0"

**xarray import errors**

Install or upgrade xarray:

.. code-block:: bash

    pip install --upgrade "xarray>=0.16.0"

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/daniloceano/MagicA/issues>`_
2. Review the :doc:`tutorials/index` for examples
3. Consult the :doc:`api/core` documentation
4. Open a new issue on GitHub with details about your problem

Next Steps
----------

After installation:

* Start with the :doc:`quickstart` guide
* Explore the :doc:`tutorials/index`
* Read about :doc:`api/core` and :doc:`api/auto_fitter`
* Check out the :doc:`tutorials/magic_adjuster_tutorial` for comprehensive examples
