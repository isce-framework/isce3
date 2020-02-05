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
* :doc:`Projection <./core/Projection>`

I/O Datastructures
------------------

* :doc:`Raster <./io/Raster>`
* :doc:`IH5File <./io/IH5>`

Product Datastructures
----------------------
* :doc:`RadarGridParameters <./product/RadarGridParameters>`
* :doc:`Product <./product/Product>`
* :doc:`Metadata <./product/Metadata>`

Image Datastructures
--------------------

* :doc:`ResampSlc <./image/ResampSlc>`

Geometry Datastructures
------------------------
* :doc:`Rdr2geo <./geometry/Rdr2geo>`
* :doc:`Geo2rdr <./geometry/Geo2rdr>`
* :doc:`DEMInterpolator <./geometry/DEMInterpolator>`

Geometry Functions
---------------------
* :doc:`getGeoPerimeter <./geometry/getGeoPerimeter>`
* :doc:`rdr2geo_point <./geometry/Rdr2geo_pt>`
* :doc:`geo2rdr_point <./geometry/Geo2rdr_pt>`

