:orphan:

.. title:: pyPoly1d

pyPoly1d
========

pyPoly1d is a data structure meant to capture 1D functions of the form

.. math::

   f(x) = \sum_{k=0}^{N} c_k \cdot \left( \frac{x-\mu}{\sigma} \right)^k 

where

* :math:`N` is the order of the polynomial.
* :math:`\mu` is the mean.
* :math:`\sigma` is the norm.
* :math:`[c_0, c_1, ..., c_N]` is the set of coefficients.

.. autoclass:: isceextension.pyPoly1d
    :members:
