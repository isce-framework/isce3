:orphan:

.. title:: Image Resampling Tutorial

Image Resampling Tutorial
=========================

One of the key operations for SLC co-registration is resampling SLC images from one geometry to another. Resampling is the process of moving pixels from one place in an image to another in a new image while accounting for fractional pixel indices via interpolation. Currently, we represent the transformation with a pixel-by-pixel map of "range" offsets (i.e., offsets in the horizontal direction) and "azimuth" offsets (i.e., offsets in the vertical direction). For SLC co-registration, in addition to resampling of the complex pixel data, the user may also need to account for non-zero carrier phases in the azimuth direction (e.g., for native Doppler images) and flattening of the complex phase to account for differences in center frequency between the image and a master image. The isce::image::ResampSlc class contains all relevant operations for SLC co-registration.

Example 1: SLC resampling
=========================

For the basic task of image resampling without carrier phase or flattening considerations, let's look at the following example. Here, we have a crop of an Envisat SLC, azimuth offsets that represent a contraction of the image in the vertical dimension, and range offsets that represent a shearing in the horizontal dimension.

.. figure:: /../doxygen/Figures/resamp_demo.png
   :scale: 100%
   :align: center
   :figclass: align-center

The following is example code to perform the resampling.

.. code-block:: python

    import isceextension
    import gdal
    
    # Instantiate a ResampSlc object
    resamp = isceextension.pyResampSlc()
    
    # Open rasters for input files
    inputSlc = isceextension.pyRaster('input.slc')
    rgOff = isceextension.pyRaster('range.off')
    azOff = isceextension.pyRaster('azimuth.off')
    
    # Create raster for output resampled SLC
    outputSlc = isceextension.pyRaster('output.slc', access=1, width=rgOff.width,
                                       length=rgOff.length, numBands=1, driver='ISCE',
                                       dtype=gdal.GDT_CFloat32)
    
    # Run resamp
    resamp.resamp(inSlc=inputSlc, outSlc=outputSlc, rgoffRaster=rgOff, azoffRaster=azOff)

First, we created pyRaster objects for all input rasters: the input SLC image, the pixel-by-pixel range offsets, and the pixel-by-pixel azimuth offsets. We then created an output SLC image with the output geometry determined by the either the range or azimuth offset raster. These objects are then passed to a default pyResampSlc instance to perform the resampling. After resampling, we obtain the following image:

.. figure:: /../doxygen/Figures/resamp_demo_result.png
   :scale: 100%
   :align: center
   :figclass: align-center
                                      
Example 2: SLC resampling with carrier phase and flattening
===========================================================

If the user wishes to account for SLCs with native Doppler frequencies in the azimuth direction and flattening with respect to a master SLC, the following code can be used.

.. code-block:: python

    import isceextension
    import gdal
    
    # Create a polynomial for native Doppler
    # Note: 0th order in azimuth, 2nd order in range
    doppler = isceextension.pyPoly2d(azimuthOrder=0, rangeOrder=2)
    doppler.coeffs = [301.35306906319204, -0.04633312447837377, 2.044436266418998e-06]
    
    # Create an ImageMode for the input image using Envisat parameters
    mode = isceextension.pyImageMode()
    # Set relevant parameters
    mode.wavelength = 0.056
    mode.startingRange = 826988.69
    mode.rangePixelSpacing = 7.80
    mode.prf = 1652.416
    
    # Create an ImageMode for the reference master image
    modeRef = isceextension.pyImageMode()
    # Set relevant parameters for reference
    modeRef.wavelength = 0.057
    modeRef.startingRange = 826991.0
    modeRef.rangePixelSpacing = 7.80
    modeRef.prf = 1652.416
    
    # Instantiate a ResampSlc object
    resamp = isceextension.pyResampSlc(doppler=doppler, mode=mode)
    resamp.refImageMode = modeRef
    
    # Open rasters for input files
    inputSlc = isceextension.pyRaster('input.slc')
    rgOff = isceextension.pyRaster('range.off')
    azOff = isceextension.pyRaster('azimuth.off')
    
    # Create raster for output resampled SLC
    outputSlc = isceextension.pyRaster('output.slc', access=1, width=rgOff.width,
                                       length=rgOff.length, numBands=1, driver='ISCE',
                                       dtype=gdal.GDT_CFloat32)
    
    # Run resamp
    resamp.resamp(inSlc=inputSlc, outSlc=outputSlc, rgoffRaster=rgOff, azoffRaster=azOff)

Example 3: SLC resampling with NumPy arrays
===========================================

As we saw in the raster tutorials, rasters can be created from NumPy arrays via the GDAL gdal_array class. Therefore, range and azimuth offset arrays can be created in memory, dressed as pyRaster objects, and passed to pyResampSlc as before.

.. code-block:: python

    import isceextension
    import numpy as np
    from osgeo import gdal_array
    import gdal
    
    # Instantiate a ResampSlc object
    resamp = isceextension.pyResampSlc()
    
    # Open rasters for input files
    inputSlc = isceextension.pyRaster('input.slc')
    
    # Meshgrids for coordinates for offsets
    xvals = np.arange(500)
    yvals = np.arange(500)
    X, Y = np.meshgrid(xvals, yvals)
    
    # Create offsets
    roff = (Y - 250.0) / 5.0
    aoff = (Y - 250.0) / 2.5
    
    # Dress offset numpy arrays with gdalarray
    roff_ds = gdal_array.OpenNumPyArray(roff)
    aoff_ds = gdal_array.OpenNumPyArray(aoff)
    
    # Pass gdal datasets to pyRaster
    roff_raster = isceextension.pyRaster('', dataset=roff_ds)
    aoff_raster = isceextension.pyRaster('', dataset=aoff_ds)
    
    # Create raster for output resampled SLC
    outputSlc = isceextension.pyRaster('output.slc', access=1, width=roff_raster.width,
                                       length=roff_raster.length, numBands=1, driver='ISCE',
                                       dtype=gdal.GDT_CFloat32)
    
    # Run resamp
    resamp.resamp(inSlc=inputSlc, outSlc=outputSlc, rgoffRaster=roff_raster,
                  azoffRaster=aoff_raster)


