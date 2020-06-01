#-*- coding: utf-8 -*-
#

# The extensions
from .. import isceextension

def geocodeSlc(gslc_raster, slc_raster, dem_raster,
                      radar_grid, geo_grid,
                      orbit,
                      native_doppler, image_grid_doppler,
                      ellipsoid,
                      thresholdGeo2rdr, numiterGeo2rdr,
                      linesPerBlock, demBlockMargin,
                      flatten):
        
    """
    Wrapper for pygeocodeSlc function.
    """
    isceextension.pygeocodeSlc(gslc_raster, slc_raster, dem_raster,
                      radar_grid, geo_grid,
                      orbit,
                      native_doppler, image_grid_doppler,
                      ellipsoid,
                      thresholdGeo2rdr, numiterGeo2rdr,
                      linesPerBlock, demBlockMargin,
                      flatten)

    return None
