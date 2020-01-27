:orphan:

.. title:: Poly1d

Poly1d
========

Poly1d is a data structure meant to capture 1D functions of the form

.. math::

   f(x) = \sum_{k=0}^{N} c_k \cdot \left( \frac{x-\mu}{\sigma} \right)^k 

where

* :math:`N` is the order of the polynomial.
* :math:`\mu` is the mean.
* :math:`\sigma` is the norm.
* :math:`[c_0, c_1, ..., c_N]` is the set of coefficients.


Factory
----------

.. code-block:: python

   from isce3.core import poly1d

   obj = poly1d(**kwds)


Documentation
----------------


.. autoclass:: isce3.core.Poly1d.Poly1d
    :members:
    :inherited-members:
