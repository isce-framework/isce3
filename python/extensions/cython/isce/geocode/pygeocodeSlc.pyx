from geocodeSlc cimport *
from cython.operator cimport dereference as deref

def pygeocodeSlc(pyRaster outputGSlc,
        pyRaster inputRSlc,
        pyRaster dem,
        pyRadarGridParameters radarGrid,
        pyGeoGridParameters geoGrid,
        pyOrbit orbit,
        pyLUT2d nativeDoppler,
        pyLUT2d imageGridDoppler,
        pyEllipsoid ellipsoid,
        double thresholdGeo2rdr,
        int numiterGeo2rdr,
        size_t linesPerBlock,
        double demBlockMargin,
        bool flatten):

    

    geocodeSlc(deref(outputGSlc.c_raster),
            deref(inputRSlc.c_raster),
            deref(dem.c_raster),
            deref(radarGrid.c_radargrid),
            deref(geoGrid.c_geogrid),
            orbit.c_orbit,
            deref(nativeDoppler.c_lut),
            deref(imageGridDoppler.c_lut),
            deref(ellipsoid.c_ellipsoid),
            thresholdGeo2rdr,
            numiterGeo2rdr,
            linesPerBlock,
            demBlockMargin,
            flatten)

    return 
