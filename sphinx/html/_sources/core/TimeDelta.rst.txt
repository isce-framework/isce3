:orphan:

.. title:: TimeDelta

TimeDelta
===========

TimeDelta is meant to represent time-intervals with double precision sub-second accuracy. There is no loss of precision (at nano-second scale) as long as the intervals represented are less than ~14 days.


Factory
----------

.. code-block:: python
   
   from isce3.core import timeDelta

   obj = timeDelta(**kwds)


Documentation
----------------

.. autoclass:: isce3.core.TimeDelta.TimeDelta
   :members:
   :inherited-members:
