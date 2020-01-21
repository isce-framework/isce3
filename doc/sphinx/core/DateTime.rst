:orphan:

.. title:: DateTime

DateTime
==========

DateTime is meant to represent a date time tag with double precision sub-second accuracy.

* The class also includes support for some time tag manipulation via the + and - operators. 
* The class also implements a rich comparison special function for comparison of time tags.

Factory
----------

.. code-block:: python
   
   from isce3.core import dateTime

   obj = dateTime(**kwds)


Documentation
---------------

.. autoclass:: isce3.core.DateTime.DateTime
   :members:
   :inherited-members:
   :special-members: __add__,__sub__

