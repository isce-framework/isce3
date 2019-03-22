set (PXDS GDAL.pxd
          IH5.pxd
          pyGDAL.pxd
          pyIH5.pxd
          pyRaster.pxd
          Raster.pxd)

add_isce_cythondir(io "${PXDS}")

