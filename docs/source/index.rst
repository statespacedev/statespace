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
    sudo apt-get -qq install -y python3.6 python3-venv python3-pip
    git clone git@gitlab.com:noahhsmith/statespace.git statespace
    cd statespace
    python3 -m venv venv
    . venv/bin/activate
    python3 setup.py develop
    pytest
    python3 statespace --demo

processors
==================================================================================================================

api.py
--------

.. automodule:: libstatespace_.api
    :members:

classical.py
---------------

baseline extended kalman filters. both the standard textbook form, and the ud factorized form.

.. automodule:: statespace.classical
    :members:

modern.py
-----------

sigma-point sampling kalman filters.

.. automodule:: statespace.modern
    :members:

particle.py
-------------

particle filters, sequential monte carlo sampling processors.

.. automodule:: statespace.particle
    :members:

models
==================================================================================================================

will probably be moving towards a higher-level statespace model, encompassing specific lower-level models - possibly something involoving a translator / converter / adaptor... the models here are already an extrememly primitive form of that - making them as similar as possible from the perspective of the classical, modern, particle processors. we can think about these becoming specific cases of something more fundamental.

modelbase.py
-------------

placeholder for what could grow to become a higher-level statespace model - with individual cases inheriting and overriding.

.. automodule:: models.modelbase
    :members:

onestate.py
--------------

.. automodule:: models.onestate
    :members:

threestate.py
--------------

.. automodule:: models.threestate
    :members:

bearingsonly.py
----------------

the bearings only problem has some interesting history - it's basically about being on a ww2 era sub. your sub is travelling along, and you begin hearing the sound of a ship at some bearing. over time, as the bearing changes, you can estimate the position and velocity of the ship. keep in mind that both your sub and the ship are moving the entire time. at some point you make a course change for your sub to pursue the ship.

.. automodule:: models.bearingsonly
    :members:

