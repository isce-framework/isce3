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

    def __cinit__(self, pyRadarGridParameters radarGrid,
                  pyLUT2d doppler,
                  double wavelength,
                  pyRadarGridParameters referenceRadarGrid=None,
                  double referenceWavelength=0.0):

        if referenceRadarGrid is not None and referenceWavelength!=0.0:
            print("no referencing")
            # constructor for flattening resampled SLC
            self.c_resamp = new ResampSlc(deref(doppler.c_lut), 
                    radarGrid.startingRange, radarGrid.rangePixelSpacing, 
                    radarGrid.sensingStart, radarGrid.prf, 
                    wavelength,
                    referenceRadarGrid.startingRange, 
                    referenceRadarGrid.rangePixelSpacing,
                    referenceWavelength)
        else:
            # constructor for not flattening resampled SLC
            self.c_resamp = new ResampSlc(deref(doppler.c_lut), 
                    radarGrid.startingRange, radarGrid.rangePixelSpacing, 
                    radarGrid.sensingStart, radarGrid.prf, 
                    wavelength)

        self.__owner = True

        # Bind the C++ LUT2d class to the Cython pyLUT2d instance
        self.py_doppler = pyLUT2d()
        del self.py_doppler.c_lut
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
        cdef string freq_str = pyStringToBytes(freq)
        self.c_resamp.referenceProduct(deref(refProduct.c_product), freq_str[0])

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
        del self.py_doppler.c_lut
        self.py_doppler.c_lut = &self.c_resamp.doppler()

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
               infile=None, outfile=None, rgfile=None, azfile=None,
               int inputBand=1, bool flatten=True, bool isComplex=True,
               int rowBuffer=40):
        """
        Run resamp on complex image data stored in HDF5 product or specified
        as an external file.

        Args:
            inSlc (Optional[pyRaster]):             Input SLC raster.
            outSlc (Optional[pyRaster]):            Output SLC raster.
            rgoffRaster (Optional[pyRaster]):       Input range offset raster.
            azoffRaster (Optional[pyRaster]):       Input azimuth offset raster.
            infile (Optional[str]):                 Filename for input SLC raster.
            outfile (Optional[str]):                Filename for output resampled image.
            rgfile (Optional[str]):                 Filename for range offset raster.
            azfile (Optional[str]):                 Filename for azimuth offset raster.
            inputBand (Optional[int]):              Band number for external file.
            flatten (Optional[bool]):               Flatten the resampled image.
            isComplex (Optional[bool]):             Input image is complex.
            rowBuffer (Optional[int]):              Row padding for reading image tiles.

        Returns:
            None
        """
        cdef string outputFile, rgoffFile, azoffFile, inputFile
    
        if (inSlc is not None and outSlc is not None and
                rgoffRaster is not None and azoffRaster is not None):

            # Call resamp directly with rasters
            self.c_resamp.resamp(deref(inSlc.c_raster), deref(outSlc.c_raster),
                                 deref(rgoffRaster.c_raster), deref(azoffRaster.c_raster),
                                 inputBand, flatten, isComplex, rowBuffer)

        elif (infile is not None and outfile is not None and
              rgfile is not None and azfile is not None):

            # Convert Python strings to C++ compatible strings
            inputFile = pyStringToBytes(infile)
            outputFile = pyStringToBytes(outfile)
            rgoffFile = pyStringToBytes(rgfile)
            azoffFile = pyStringToBytes(azfile)
       
            # Call resamp with filenames 
            self.c_resamp.resamp(inputFile, outputFile, rgoffFile, azoffFile,
                                 inputBand, flatten, isComplex, rowBuffer)

        else:
            assert False, 'No input rasters or filenames provided'

    
# end of file
