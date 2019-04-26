from cython.operator cimport dereference as deref
from RTC cimport facetRTC

def pyRTC_impl(pyProduct prod, pyRaster in_raster, pyRaster out_raster):
    facetRTC(deref(prod.c_product),
             deref(in_raster.c_raster),
             deref(out_raster.c_raster))

# Wrapper to support output as filename or pyRaster
def pyRTC(pyProduct prod, pyRaster in_raster, out_raster):
    # Type-check output raster
    if type(out_raster) != pyRaster:
        # Create output raster if filename is given
        if type(out_raster) == str:
            filename = out_raster
            out_raster = pyRaster(filename, access=1, width=in_raster.width,
                                                      length=in_raster.length)
        else:
            raise TypeError("must pass pyRaster or filename to pyRTC")
    else:
        # Enforce output raster is writable
        if out_raster.access != 1:
            raise ValueError("output raster must be writable")

    pyRTC_impl(prod, in_raster, out_raster)
