:orphan:

.. title:: getGeoPerimeter

getGeoPerimeter
==================

This is an extension of Rdr2Geo but only performs the transformation of radar coordiantes to 
map cordinates at the edges of the swath and returns a GeoJson string representing the Perimeter
of the radar grid. The Geojson form allows one to use the result directly with other GIS software
and REST interfaces.


Documentation
---------------

.. autofunction:: isce3.geometry.getGeoPerimeter
