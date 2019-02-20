#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017
#

from libcpp cimport bool
from libcpp.string cimport string
from ResampSlc cimport ResampSlc

cdef class pyResampSlc:
    """
    Cython wrapper for isce::image::ResampSlc.

    Args:
        product (pyProduct):                        Product to be resampled.
        refProduct (Optional[pyProduct]):           Reference product for flattening.

    Returns:
        None.
    """
    # C++ class pointer
    cdef ResampSlc * c_resamp
    cdef bool __owner

    # Cython class objects
    cdef pyLUT2d py_doppler
    
    def __cinit__(self, pyProduct product, pyProduct refProduct=None, freq='A'):
        """
        Initialize C++ objects.
        """
        if product is not None:
            self.c_resamp = new ResampSlc(deref(product.c_product),
                                          pyStringToBytes(freq))

        elif product is not None and refProduct is not None:
            self.c_resamp = new ResampSlc(deref(product.c_product),
                                          deref(refProduct.c_product),
                                          pyStringToBytes(freq))
        self.__owner = True

        # Bind the C++ LUT2d class to the Cython pyLUT2d instance
        self.py_doppler.c_lut = &self.c_resamp.doppler()
        self.py_doppler.__owner = False

        return

    def __dealloc__(self):
        if self.__owner:
            del self.c_resamp

    def setReferenceProduct(self, pyProduct refProduct, freq='A'):
        """
        Set a reference product for flattening.
        """
        self.c_resamp.referenceProduct(deref(refProduct.c_product),
                                       pyStringToBytes(freq))

    @property
    def doppler(self):
        """
        Get the content Doppler LUT for the product to be resampled.
        """
        lut = pyLUT2d.bind(self.py_doppler)
        return self.lut

    @doppler.setter
    def doppler(self, pyLUT2d dop):
        """
        Override the content Doppler LUT from a pyLUT1d instance.
        """
        self.c_resamp.doppler(deref(dop.c_lut))

    @property
    def linesPerTile(self):
        """
        Get the number of lines per processing tile.
        """
        return self.c_resamp.linesPerTile()

    @linesPerTile.setter
    def linesPerTile(self, int lines):
        """
        Set the number of lines per processing tile.
        """
        self.c_resamp.linesPerTile(lines)

    def resamp(self, pyRaster inSlc=None, pyRaster outSlc=None,
               pyRaster rgoffRaster=None, pyRaster azoffRaster=None,
               outfile=None, rgfile=None, azfile=None, pol='hh', int inputBand=1,
               bool flatten=True, bool isComplex=True, int rowBuffer=40, infile=None):
        """
        Run resamp on complex image data stored in HDF5 product or specified
        as an external file.

        Args:
            inSlc (Optional[pyRaster]):             Input SLC raster.
            outSlc (Optional[pyRaster]):            Output SLC raster.
            rgoffRaster (Optional[pyRaster]):       Input range offset raster.
            azoffRaster (Optional[pyRaster]):       Input azimuth offset raster.
            outfile (Optional[str]):                Filename for output resampled image.
            rgfile (Optional[str]):                 Filename for range offset raster.
            azfile (Optional[str]):                 Filename for azimuth offset raster.
            pol (Optional[str]):                    Polarization. Default: hh.
            inputBand (Optional[int]):              Band number for external file.
            flatten (Optional[bool]):               Flatten the resampled image.
            isComplex (Optional[bool]):             Input image is complex.
            rowBuffer (Optional[int]):              Row padding for reading image tiles.

        Returns:
            None
        """
        cdef string outputFile, rgoffFile, azoffFile, polarization, inputFile
    
        if (inSlc is not None and outSlc is not None and
                rgoffRaster is not None and azoffRaster is not None):

            # Call resamp directly with rasters
            self.c_resamp.resamp(deref(inSlc.c_raster), deref(outSlc.c_raster),
                                 deref(rgoffRaster.c_raster), deref(azoffRaster.c_raster),
                                 inputBand, flatten, isComplex, rowBuffer)

        elif outfile is not None and rgfile is not None and azfile is not None:

            # Convert Python strings to C++ compatible strings
            outputFile = pyStringToBytes(outfile)
            rgoffFile = pyStringToBytes(rgfile)
            azoffFile = pyStringToBytes(azfile)
        
            # Call correct resamp signature
            if infile is None:
                polarization = pyStringToBytes(pol.lower())
                self.c_resamp.resamp(outputFile, polarization, rgoffFile, azoffFile,
                                     flatten, isComplex, rowBuffer)
            else:
                inputFile = pyStringToBytes(infile)
                self.c_resamp.resamp(inputFile, outputFile, rgoffFile, azoffFile,
                                     inputBand, flatten, isComplex, rowBuffer)

        else:
            assert False, 'No input rasters or filenames provided'

    
# end of file
