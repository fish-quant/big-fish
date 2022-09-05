.. big-fish documentation master file, created by
   sphinx-quickstart on Thu Nov 19 22:45:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Big-FISH
********

Getting started
===============

To avoid dependency conflicts, we recommend the use of a dedicated
`virtual <https://docs.python.org/3.6/library/venv.html>`_ or `conda
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-
environments.html>`_ environment. In a terminal run the command:

.. code-block:: bash

   $ conda create -n bigfish_env python=X.Y
   $ source activate bigfish_env

With X.Y a valid Python version greater or equal than 3.6. Note that Big-FISH
is tested only for Python 3.6, 3.7, 3.8 and 3.9.

We then recommend two options to install Big-FISH in your virtual environment:
from PyPi or GitHub.

Download the package from PyPi
------------------------------

Use the package manager `pip <https://pip.pypa.io/en/stable>`_ to install
Big-FISH. In a terminal run the command:

.. code-block:: bash

   $ pip install big-fish

Clone package from GitHub
-------------------------

Clone the project's `GitHub repository <https://github.com/fish-quant/big-
fish>`_ and install it manually with the following commands:

.. code-block:: bash

   $ git clone git@github.com:fish-quant/big-fish.git
   $ cd big-fish
   $ pip install .

------------

Examples
========

Several examples are available as `Jupyter notebooks <https://github.com/fish-
quant/big-fish-examples/tree/master/notebooks>`_:

#. Read and write images.
#. Normalize and filter images.
#. Project in two dimensions.
#. Segment nuclei and cells.
#. Detect spots.
#. Extract cell level results.
#. Analyze coordinates.

To run these notebooks, you will need to clone the notebook repository:

.. code-block:: bash

   $ git clone git@github.com:fish-quant/big-fish-examples.git

Activate your environment and install Big-FISH and Jupyter notebook dependencies inside:

.. code-block:: bash

   $ source activate bigfish_env
   $ cd big-fish-examples
   $ pip install .

Then launch the notebooks:

.. code-block:: bash

   $ jupyter notebook

You can also run these example online with `mybinder <https://mybinder.org/v2/
gh/fish-quant/fq-imjoy/binder?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252F
github.com%252Ffish-quant%252Fbig-fish-examples%26urlpath%3Dtree%252Fbig-fish-
examples%252Fnotebooks%26branch%3Dmaster>`_. The remote server can take a bit
of time to start.

------------

API reference
*************

.. toctree::
   :caption: Preprocessing

   stack/io
   stack/preprocessing
   stack/augmentation

------------

.. toctree::
   :caption: Spot detection

   detection/spots
   detection/dense
   detection/subpixel
   detection/cluster
   detection/colocalization

------------

.. toctree::
   :caption: Segmentation

   segmentation/nucleus
   segmentation/cell
   segmentation/postprocessing

------------

.. toctree::
   :caption: Analysis

   classification/extraction
   classification/features

------------

.. toctree::
   :caption: Visualization

   plot/plot_image
   plot/plot_detection
   plot/plot_segmentation
   plot/plot_coordinate

------------

.. toctree::
   :caption: Utils

   utils/utils

------------

Support
=======

If you have any question relative to the package, please open an `issue
<https://github.com/fish-quant/big-fish/issues>`_ on Github.

------------

Citation
========

If you exploit this package for your work, please cite:

.. code-block:: text

    Arthur Imbert, Wei Ouyang, Adham Safieddine, Emeline Coleno, Christophe
    Zimmer, Edouard Bertrand, Thomas Walter, Florian Mueller. FISH-quant v2:
    a scalable and modular analysis tool for smFISH image analysis. bioRxiv
    (2021) https://doi.org/10.1101/2021.07.20.453024
