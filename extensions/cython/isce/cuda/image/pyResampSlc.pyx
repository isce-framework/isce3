#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017
#

from libcpp cimport bool
from libcpp.string cimport string
from cython.operator cimport dereference as deref

# Pull in Cython classes from isceextension
from isceextension cimport pyLUT1d
from isceextension cimport pyProduct
from isceextension cimport pyImageMode
from isceextension cimport pyRaster

from cuResampSlc cimport ResampSlc

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
    
    def __cinit__(self, product=None, doppler=None, mode=None):
        """
        Initialize C++ objects.
        """
        cdef pyProduct c_product
        cdef pyLUT1d c_doppler
        cdef pyImageMode c_mode
        
        if product is not None:
            c_product = <pyProduct> product
            self.c_resamp = new ResampSlc(deref(c_product.c_product))

        elif doppler is not None and mode is not None:
            c_doppler = <pyLUT1d> doppler
            c_mode = <pyImageMode> mode
            self.c_resamp = new ResampSlc(deref(c_doppler.c_lut), deref(c_mode.c_imagemode))

        else:
            self.c_resamp = new ResampSlc()

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
        Get the content Doppler LUT for the product to be resampled.
        """
        lut = pyLUT1d.cbind(self.c_resamp.doppler())
        return self.lut

    @doppler.setter
    def doppler(self, pyLUT1d dop):
        """
        Override the content Doppler LUT from a pyLUT1d instance.
        """
        self.c_resamp.doppler(deref(dop.c_lut))

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
