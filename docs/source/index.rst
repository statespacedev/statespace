.. toctree::
   :maxdepth: 2
   :caption: Contents:

introduction
==================================================================================================================

this project focuses on reference problems in kalman, sigma-point, and particle processors - in particular, numpy and eigen implementations, using classic vector and matrix representations so things look like what's in the literature.

install
------------------------------------------------------------------------------------------------------------------

have switched to a container build environment - reproducible using the dockerfile. everything needed is there and can be reproduced for a local environment. docker image is available on docker hub.

processors
==================================================================================================================

kalman.py
---------------

baseline extended kalman filters. both the standard textbook form, and the ud factorized form.

.. automodule:: statespace.processors.kalman
    :members:

sigmapoint.py
----------------

sigma-point sampling kalman filters. this is the ukf or unscented kalman filter, or 'modern' kalman filtering. it's essentially somewhere between a classical kalman filter, where uncertainty is represented as gaussian, and a particle filter, where uncertainty has arbitrary shape. here uncertainty is deterministically sampled at a small number of points, the sigma points or sigma particles. in a particle filter the number and role of the particles are increased.

.. automodule:: statespace.processors.sigmapoint
    :members:

particle.py
-------------

particle filters, sequential monte carlo sampling processors. sampling here is random, not deterministic as in the sigmapoint processor. and the idea of resampling and growing new particles comes to the fore. the particles are random and new ones can be introduced freely.

.. automodule:: statespace.processors.particle
    :members:

models
==================================================================================================================

will probably be moving towards a higher-level statespace model, encompassing specific lower-level models - possibly something involving a translator / converter / adaptor... the models here are already an extrememly primitive form of that - making them as similar as possible from the perspective of the classical, modern, particle processors. we can think about these becoming specific cases of something more fundamental.

basemodel.py
-------------

placeholder for what could grow to become a higher-level statespace model - with individual models inheriting and overriding.

.. automodule:: statespace.models.basemodel
    :members:

onestate.py
--------------

a simple as possible one-state example with non linear temporal and observation updates. it's a common example in the candy and jazwinisky books. based on real world reentry vehicle tracking.

.. automodule:: statespace.models.onestate
    :members:

threestate.py
--------------

three-state extension of the the one-state model. non linear temporal and observation updates.

.. automodule:: statespace.models.threestate
    :members:

bearingsonly.py
----------------

the bearings only problem has some interesting history. it's basically about being on a sub. your sub is travelling along steadily and you begin hearing the sound of a ship at some bearing. over time and as the bearing changes, you can estimate the relative position and velocity of the ship. at some point you make a course change for your sub to pursue the ship.

.. automodule:: statespace.models.bearingsonly
    :members:

