:orphan:

.. title:: Raster Tutorial

Raster Tutorial
===============

ISCE delegates raster data handling to the `Geospatial Data Abstraction Library (GDAL) <https://gdal.org/>`_ at the C++ level via isce::io::Raster. Even the Python interface to isce::io::Raster (pyRaster) is designed to work in combination with GDAL's Python bindings. We will walk through some of the examples here. 

A concise introduction to GDAL's Python bindings can be found `here <https://www.gdal.org/gdal_tutorial.html>`_. Familiarizing yourself with this will help navigate the following tutorial more easily. Note that pyRaster is a thin wrapper over GDAL Dataset and is only needed when you want to pass on the created object to processing modules in ISCE. For simple I/O at the python level, one should directly use the GDAL interface.

This tutorial is organized in 3 sections.

1. :ref:`readonlymode`
2. :ref:`updatemode`
3. :ref:`createmode`
4. :ref:`createnumpy`

.. _readonlymode:

Open an existing file in Read-Only Mode
---------------------------------------


.. code-block:: python

   from isceextension import pyRaster

   #Create object
   raster = pyRaster(inputfilename)
   print('Dims: {0}P x {1}L'.format(raster.width, raster.length))


.. _updatemode:

Open an existing file in Update Mode
------------------------------------


.. code-block:: python

   from isceextension import pyRaster
   from osgeo import gdal

   #Create object
   raster = pyRaster(inputfilename, gdal.GA_Update)
   for ii in range(raster.numBands):
      print('Band {0} is of type {1}'.format(ii+1, gdal.GetDataTypeName(raster.getDataType(ii+1))))

.. _createmode:

Creating a raster
-----------------

It is really easy to create raster with GDAL and then pass it to ISCE to work with.

.. code-block:: python

   from isceextension import pyRaster
   from osgeo import gdal

   #Create GDAL raster
   driver = gdal.GetDriverByName('GTiff')
   ds = driver.Create("antarctica.tif", 1000, 1500, 2, gdal.GDT_Float32);
   ds.SetGeoTransform([-1.0e6, 1.0e3, 0., 1.5e6, 0., -1.0e3])

   #Wrap it with pyRaster
   raster = pyRaster('', dataset=ds)

   #Set projection code. Can do this with GDAL+osr as well before creating raster.
   raster.EPSG = 3031


.. _createnumpy:

Creating a raster using numpy array
-----------------------------------

You can also create GDAL datasets out of numpy arrays and pass it to ISCE to work with. 

.. code-block:: python

   from isceextension import pyRaster
   from osgeo import gdal_array
   import numpy as np

   #Create numpy array
   arr = np.ones((1500,1000), dtype=np.complex64)

   #Dress numpy array with gdalarray
   ds = gdal_array.OpenNumPyArray(arr, gdal.GA_Update)

   #Pass gdal dataset to pyRaster
   raster = pyRaster('', dataset=ds)
