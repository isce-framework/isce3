:orphan:

.. title:: Raster Tutorial

Raster Tutorial
===============

ISCE delegates raster data handling to the `Geospatial Data Abstraction Library (GDAL) <https://gdal.org/>`_ at the C++ level via isce::io::Raster. Even the Python interface to isce::io::Raster is designed to work in combination with GDAL's Python bindings. We will walk through some of the examples here. 

A concise introduction to GDAL's Python bindings can be found `here <https://www.gdal.org/gdal_tutorial.html>`_. Familiarizing yourself with this will help navigate the following tutorial more easily. Note that pyRaster is a thin wrapper over GDAL Dataset and is only needed when you want to pass on the created object to processing modules in ISCE. For simple I/O at the python level, one should directly use the GDAL interface.

1. :ref:`readonlymode`
2. :ref:`updatemode`
3. :ref:`createmode`
4. :ref:`createnumpy`
5. :ref:`withh5py`


.. _readonlymode:

Open an existing file in Read-Only Mode
---------------------------------------


.. code-block:: python

   from isce3.io import raster

   #Create object
   image = raster(filename="inputfilename")
   print('Dims: {0}P x {1}L'.format(raster.width, raster.length))

   #image is ready to be passed on to ISCE processing modules

.. _updatemode:

Open an existing file in Update Mode
------------------------------------


.. code-block:: python

   from isce3.io import raster
   from osgeo import gdal

   #Create object
   ds = gdal.Open(inputfilename, gdal.GA_Update)
   image = raster(dataset=ds)
   for ii in range(image.numBands):
      print('Band {0} is of type {1}'.format(ii+1, gdal.GetDataTypeName(image.getDatatype(ii+1))))

    #image is ready to be passed on to ISCE processing modules

.. _createmode:

Creating a raster
-----------------

It is really easy to create raster with GDAL and then pass it to ISCE to work with.

.. code-block:: python

   from isce3.io import raster
   from osgeo import gdal

   #Create GDAL raster
   driver = gdal.GetDriverByName('GTiff')
   ds = driver.Create("antarctica.tif", 1000, 1500, 2, gdal.GDT_Float32);
   ds.SetGeoTransform([-1.0e6, 1.0e3, 0., 1.5e6, 0., -1.0e3])

   #Wrap it with pyRaster
   image = raster(dataset=ds)

   #Set projection code. Can do this with GDAL+osr as well before creating raster.
   image.EPSG = 3031

   #image is ready to be passed on to ISCE processing modules


.. _createnumpy:

Creating a raster using numpy array
-----------------------------------

You can also create GDAL datasets out of numpy arrays and pass it to ISCE to work with. 

.. code-block:: python

   from isce3.io import raster
   from osgeo import gdal_array
   import numpy as np

   #Create numpy array
   arr = np.ones((1500,1000), dtype=np.complex64)

   #Dress numpy array with gdalarray
   ds = gdal_array.OpenArray(arr)

   #Pass gdal dataset to pyRaster
   image = raster(dataset=ds)

   #image is ready to be passed on to ISCE processing modules


.. _withh5py:

Creating a raster using h5py
----------------------------

You can also create ISCE Rasters out of h5py datasets. Note that for read only operations, you can use GDAL's 
<a href="https://gdal.org/drivers/raster/hdf5.html">HDF5 driver</a> as well and set up rasters as shown above. 

.. code-block:: python

   from isce3.io import raster 
   import h5py

   #Create HDF5 file
   fid = h5py.File('example.h5', 'w')

   #Create group
   grp = fid.create_group('level1/level2')

   #Create dataset
   #All complicated creation options can be controlled via h5py if needed
   dset = grp.create_dataset("data", shape=(100,150), dtype='f4')

   #Wrap it with ISCE raster 
   image = raster(h5=dset)

   #image is ready to be passed on to ISCE processing modules

