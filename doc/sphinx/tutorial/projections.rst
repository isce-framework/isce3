:orphan:

.. title:: Projections Tutorial

Map Projections Tutorial
========================

ISCE's Projection support is designed using `PROJ <https://proj4.org>`_ as a reference. 
We expose ISCE's projection library under isce3.core.Projection but this is meant to only inform
processing modules of projection choice. We encourage users to use 
`pyproj <https://jswhit.github.io/pyproj/>`_ or GDAL's Python bindings itself
to transform coordinates at the Python level.

For this tutorial, we assume a minimum version of 2.4 for pyProj and 3.0 for GDAL.

We will present simple examples using GDAL Python bindings and pyproj to accomplish the same 
transformations as the ones performed in the C++ tutorial. The tutorial is organized as follows:


1. :ref:`projcreate`
2. :ref:`coordinatetransformgdal`
3. :ref:`coordinatetransformpyproj`


.. _projcreate:

Create a Projection object
--------------------------


.. code-block:: python

   #Using GDAL
   from osgeo import osr
   llh = osr.SpatialReference()
   llh.ImportFromEPSG(4326)

   utm20n = osr.SpatialReference()
   utm20n.ImportFromEPSG(32620)


   #Using pyproj
   import pyproj
   llh = pyproj.Proj(4326)
   utm20n = pyproj.Proj(32620)

.. _coordinatetransformgdal:

Coordinate transformation using GDAL
------------------------------------

Note that internally, ISCE uses Lon/Lat/Hae (rad) as common projection system that ties all the different 
transformers together. At the Python level, we don't need to expose this to the user. GDAL provides methods
that are similar to isce::core::projTransform method.


Coordinate transformation using GDAL can be accomplished as follows:

.. code-block:: python

   from osgeo import osr
   import numpy as np
   
   llh = osr.SpatialReference() #Create Lat/Lon
   llh.ImportFromEPSG(4326)

   spolar = osr.SpatialReference()  #Create Antarctic Polar Stereographic
   spolar.ImportFromEPSG(3031)

   #Create transformer for Lon/Lat to Polar
   trans = osr.CoordinateTransformation(llh, spolar)


   #Transform point / Transform points
   #GDAL respects axis order of EPSG definitions
   # So use Lat/Lon/Hgt for GDAL
   x,y,z = trans.TransformPoint(-90., 0., 0.)


.. _coordinatetransformpyproj:

Coordinate transformation using pyproj
--------------------------------------

pyproj exposes forward, inverse and transformation between coordinates. The forward and inverse methods in general
only expose the horizontal datum transformations. Since all the supported coordinate systems are defined on the
WGS84 Ellipsoid, transformations should not affect the height value. Coordinate transformation using pyproj
can be accomplished as follows:

.. code-block:: python

   import pyproj

   llh = pyproj.Proj(4326)
   spolar = pyproj.Proj(3031)

   #convert Lon/Lat (deg) to Polar Stereographic (Horizontal datum)
   #pyproj always use x,y, hgt - so use Lon/Lat/Hgt
   x,y = spolar(0., -90.0)

   #convert Polar Stereographic to Lon/Lat (deg)
   lon, lat = spolar(0., 0., inverse=True)


   #For full 3D conversions using Transform
   #The transformation pipeline respects axis order
   x,y,z = pyproj.transform(llh, spolar, -90., 0., z=100.)
   lat,lon,h = pyproj.transform(spolar, llh, 0., 0., z=200.)
