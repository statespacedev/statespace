.. toctree::
   :maxdepth: 2
   :caption: Contents:

introduction
==================================================================================================================

this project focuses on reference problems in kalman, sigma-point, and particle processors - in particular, numpy and eigen implementations.

install
------------------------------------------------------------------------------------------------------------------

build-test-deploy to `pypi <https://pypi.org/project/statespace>`__ is mostly a placeholder, ubuntu clone-install-develop of `gitlab repo <https://gitlab.com/noahhsmith/statespace>`__ is assumed for now.

.. code-block:: bash

    sudo apt-get -qq update -qy
    sudo apt-get -qq install -y python3 python3-venv python3-pip
    git clone git@gitlab.com:noahhsmith/statespace.git statespace
    cd statespace
    python3 -m venv venv
    . venv/bin/activate
    python3 setup.py develop

processors
==================================================================================================================

kalman.py
---------------

baseline extended kalman filters. both the standard textbook form, and the ud factorized form.

.. automodule:: processors.kalman
    :members:

sigmapoint.py
----------------

sigma-point sampling kalman filters. this is the ukf or unscented kalman filter, or 'modern' kalman filtering. it's essentially somewhere between a classical kalman filter, where uncertainty is represented as gaussian, and a particle filter, where uncertainty has arbitrary shape. here uncertainty is sampled at a small number of points, the sigma points or sigma particles. in a particle filter the number and role of the particles are increased.

.. automodule:: processors.sigmapoint
    :members:

particle.py
-------------

particle filters, sequential monte carlo sampling processors.

.. automodule:: processors.particle
    :members:

api.py
--------

entry point into cpp implementations for use by python code.

.. automodule:: libstatespace_.api
    :members:

models
==================================================================================================================

will probably be moving towards a higher-level statespace model, encompassing specific lower-level models - possibly something involoving a translator / converter / adaptor... the models here are already an extrememly primitive form of that - making them as similar as possible from the perspective of the classical, modern, particle processors. we can think about these becoming specific cases of something more fundamental.

basemodel.py
-------------

placeholder for what could grow to become a higher-level statespace model - with individual models inheriting and overriding.

.. automodule:: models.basemodel
    :members:

onestate.py
--------------

a simple as possible one-state example with non linear temporal and observation updates. it's a common example in the candy and jazwinisky books. based on real world reentry vehicle tracking.

.. automodule:: models.onestate
    :members:

threestate.py
--------------

three-state extension of the the one-state model. non linear temporal and observation updates.

.. automodule:: models.threestate
    :members:

bearingsonly.py
----------------

the bearings only problem has some interesting history. it's basically about being on a sub. your sub is travelling along steadily and you begin hearing the sound of a ship at some bearing. over time and as the bearing changes, you can estimate the relative position and velocity of the ship. at some point you make a course change for your sub to pursue the ship.

.. automodule:: models.bearingsonly
    :members:

