.. toctree::
   :maxdepth: 2
   :hidden:

   introduction
   processors
   models
   literature

statespace
=============
this project focuses on numpy implementations of kalman, sigma point, and particle - using classic vector and matrix representations so things look like what's in the literature. needless to say, that means lots of single letter variables, with capital letters for matrices and lower case letters for vectors. *literature >> code*.

here's what numpy wants for the style of row and column vectors we're used to.

.. code-block:: python
    
    row_vector = array([[1, 2, 3]])    # shape (1, 3)
    col_vector = array([[1, 2, 3]]).T  # shape (3, 1)

we'll go with the usual here - column vectors. so the dot product of x with itself is xTx - or x.T @ x in numpy land - shapes (1, 3)(3, 1) = (1, 1). for a (5, 3) matrix A, the product Ax or A @ x is (5, 3)(3, 1) = (5, 1), and the product ATAx or A.T @ A @ x is (3, 5)(5, 3)(3, 1) = (3, 1).

we're also experimenting with going full-on docs-as-code - and with having some fun preserving some strange kind of historical vision here, as we go - jump right into the introduction and follow along as we stroll forward through the caveman ages.
