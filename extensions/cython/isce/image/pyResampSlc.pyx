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
    
    def __cinit__(self, pyProduct product):
        """
        Initialize C++ objects.
        """
        self.c_resamp = new ResampSlc(deref(product.c_product))
        self.__owner = True
        return

    def __dealloc__(self):
        if self.__owner:
            del self.c_resamp

    def setReferenceProduct(self, pyProduct refProduct):
        """
        Set a reference product for flattening.
        """
        self.c_resamp.referenceProduct(deref(refProduct.c_product))

    @property
    def doppler(self):
        """
        Get the content Doppler polynomial for the product to be resampled.
        """
        poly = pyPoly2d.cbind(self.c_resamp.doppler())
        return self.poly

    @doppler.setter
    def doppler(self, pyPoly2d dop):
        """
        Override the content Doppler polynomial from a pyPoly2d instance.
        """
        self.c_resamp.doppler(deref(dop.c_poly2d))

    @property
    def imageMode(self):
        """
        Get the image mode for the product.
        """
        mode = pyImageMode.cbind(self.c_resamp.imageMode())
        return mode

    @imageMode.setter 
    def imageMode(self, pyImageMode mode):
        """
        Override the image mode for the product.
        """
        self.c_resamp.imageMode(deref(mode.c_imagemode))

    @property
    def refImageMode(self):
        """
        Get the reference image mode for the product.
        """
        mode = pyImageMode.cbind(self.c_resamp.refImageMode())
        return mode

    @refImageMode.setter 
    def refImageMode(self, pyImageMode mode):
        """
        Override the reference image mode for the product.
        """
        self.c_resamp.refImageMode(deref(mode.c_imagemode))

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

    # Run resamp
    def resamp(self, outfile, rgfile, azfile, pol='hh', int inputBand=1,
               bool flatten=True, bool isComplex=True, int rowBuffer=40,
               infile=None):
        """
        Run resamp on complex image data stored in HDF5 product or specified
        as an external file.

        Args:
            outfile (str):                          Filename for output resampled image.
            rgfile (str):                           Filename for range offset raster.
            azfile (str):                           Filename for azimuth offset raster.
            pol (Optional[str]):                    Polarization. Default: hh.
            inputBand (Optional[int]):              Band number for external file.
            flatten (Optional[bool]):               Flatten the resampled image.
            isComplex (Optional[bool]):             Input image is complex.
            rowBuffer (Optional[int]):              Row padding for reading image tiles.

        Returns:
            None
        """
        cdef string outputFile = pyStringToBytes(outfile)
        cdef string rgoffFile = pyStringToBytes(rgfile)
        cdef string azoffFile = pyStringToBytes(azfile)
        
        # Call correct resamp routine
        cdef string polarization
        cdef string inputFile
        if infile is None:
            polarization = pyStringToBytes(pol.lower())
            self.c_resamp.resamp(outputFile, polarization, rgoffFile, azoffFile,
                                 flatten, isComplex, rowBuffer)
        else:
            inputFile = pyStringToBytes(infile)
            self.c_resamp.resamp(inputFile, outputFile, rgoffFile, azoffFile,
                                 inputBand, flatten, isComplex, rowBuffer)

        return


    def resamp_temp(self, pyRaster inSlc, pyRaster outSlc, pyRaster rgoffRaster,
                    pyRaster azoffRaster, int inputBand=1, bool flatten=True,
                    bool isComplex=True, int rowBuffer=40):


        self.c_resamp.resamp(deref(inSlc.c_raster), deref(outSlc.c_raster),
                             deref(rgoffRaster.c_raster), deref(azoffRaster.c_raster),
                             inputBand, flatten, isComplex, rowBuffer)
    
# end of file
