:orphan:

.. title:: Cython


ISCE's Cython Interface 
=======================

The cython extension module that is built from the C++ code is available at the python level as "isce3.extensions.isceextension".

The following classes are exposed via Cython:

Core Datastructures
--------------------
These are data structures that are directly related to C++ classes in the isce::core namespace

* :doc:`pyDateTime <./core/pyDateTime>` 
* :doc:`pyTimeDelta <./core/pyTimeDelta>`
* :doc:`pyEllipsoid <./core/pyEllipsoid>`
* :doc:`pyEulerAngles <./core/pyEulerAngles>`
* :doc:`pyPoly1d <./pyPoly1d>`
* :doc:`pyPoly2d <./pyPoly2d>`
* :doc:`pyLUT1d <./pyLUT1d>`
* :doc:`pyOrbit <./pyOrbit>`

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
