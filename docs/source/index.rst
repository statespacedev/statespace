.. toctree::
   :maxdepth: 2
   :caption: Contents:

introduction
==================================================================================================================

this project focuses on reference problems from Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods, Kalman Filtering: Theory and Practice, and Stochastic Processes and Filtering Theory - in particular, using numpy for matrix and vector manipulation.

install
------------------------------------------------------------------------------------------------------------------

build-test-deploy to `pypi <https://pypi.org/project/statespace>`__ is mostly a placeholder, ubuntu clone-install-develop of `gitlab repo <https://gitlab.com/noahhsmith/statespace>`__ is assumed for now.

.. code-block:: bash

    sudo apt-get -qq update -qy
    sudo apt-get -qq install -y python3.6 python3-venv python3-pip
    git clone git@gitlab.com:noahhsmith/statespace.git statespace
    cd statespace
    python3 -m venv venv
    . venv/bin/activate
    python3 setup.py develop
    pytest
    python3 statespace --demo

statespace algorithms
==================================================================================================================

classical.py
--------------

.. automodule:: statespace.classical
    :members:

modern.py
----------

.. automodule:: statespace.modern
    :members:

particle.py
-------------

.. automodule:: statespace.particle
    :members:

models
==================================================================================================================

jazwinski1.py
-------------

.. automodule:: models.jazwinski1
    :members:

jazwinski2.py
-------------

.. automodule:: models.jazwinski2
    :members:

rccircuit.py
-------------

.. automodule:: models.rccircuit
    :members:
