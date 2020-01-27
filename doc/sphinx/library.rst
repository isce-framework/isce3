:orphan:

.. title:: Cython


ISCE's Library Interface 
==========================

ISCE's library interface is meant to mimic the C++ code structure.

The following classes are exposed to the Python level:

Core Datastructures
--------------------
These are data structures that are directly related to C++ classes in the isce::core namespace

* :doc:`DateTime <./core/DateTime>` 
* :doc:`TimeDelta <./core/TimeDelta>`
* :doc:`Ellipsoid <./core/Ellipsoid>`
* :doc:`EulerAngles <./core/EulerAngles>`
* :doc:`Poly1d <./core/Poly1d>`
* :doc:`Poly2d <./core/Poly2d>`
* :doc:`LUT1d <./core/LUT1d>`
* :doc:`Orbit <./core/Orbit>`

I/O Datastructures
------------------

* :doc:`pyIH5File <./pyIH5>`
* :doc:`pyRaster <./pyRaster>`

Product Datastructures
----------------------

* :doc:`pyProduct <./pyProduct>`
* :doc:`pyMetadata <./pyMetadata>`

Image Datastructures
--------------------

* :doc:`pyResampSlc <./pyResampSlc>`

Geometry Datastructures
-----------------------
* :doc:`pyTopo <./pyTopo>`
* :doc:`pyGeo2rdr <./pyGeo2rdr>`
