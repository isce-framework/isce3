:orphan:

.. title:: pyPoly2d

pyPoly2d
========

pyPoly2d is a data structure meant to capture 1D functions of the form

.. math::

   f(y,x) = \sum_{i=0}^{N_y} \sum_{j=0}^{N_x} c_{ij} \cdot \left( \frac{y-\mu_y}{\sigma_y} \right)^i \cdot \left( \frac{x-\mu_x}{\sigma_x} \right)^j 

where

* :math:`N_y` and :math:`N_x` are the orders of the polynomial in azimuth (y) and range (x).
* :math:`\mu_y` and :math:`\mu_x` are the means.
* :math:`\sigma_y` and :math:`\sigma_x` are the norms.
* :math:`[c_{00}, c_{01}, ..., c_{N_yN_x}]` is the set of coefficients.


.. autoclass:: isceextension.pyPoly2d
   :members:
