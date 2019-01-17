#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from ImageMode cimport ImageMode, sizearray2

cdef class pyImageMode:
    """
    Cython wrapper for isce::product::ImageMode.

    Args:
        mode (Optional[str]):                   Mode ('aux', 'primary')

    Return:
        None
    """
    # C++ class pointers
    cdef ImageMode * c_imagemode
    cdef bool __owner
    
    def __cinit__(self, mode='primary'):
        """
        Constructor that creates a C++ isce::product::ImageMode objects and binds it to
        python instance.
        """
        self.c_imagemode = new ImageMode(<string> pyStringToBytes(mode))
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_imagemode

    @staticmethod
    cdef cbind(ImageMode mode):
        '''
        Bind a pyImageMode mode to a specific ImageMode pointer.
        '''
        new_mode = pyImageMode()
        del new_mode.c_imagemode
        new_mode.c_imagemode = new ImageMode(mode)
        new_mode.__owner = True
        return new_mode

    @property
    def length(self):
        """
        Get the image length.
        """
        return self.c_imagemode.length()

    @property
    def width(self):
        """
        Get the image width.
        """
        return self.c_imagemode.width()

    def setDimensions(self, dims):
        '''
        Set the dimensions of ImageMode.

        Args:
            dims (list):     Pair of integers [length, width]
        '''
        cdef sizearray2 indims
        indims[0] = dims[0]
        indims[1] = dims[1]
        self.c_imagemode.dataDimensions(indims)

    def dataPath(self, pol):
        """
        Get the HDF5 path to a dataset with a given polarization.

        Args:
            pol (str):                      Polarization ('hh', 'hv', 'vh', 'vv')

        Return:
            None
        """
        return self.c_imagemode.dataPath(<string> pyStringToBytes(pol.lower()))

    @property
    def prf(self):
        """
        Get PRF value.
        """
        return self.c_imagemode.prf()

    @prf.setter
    def prf(self, double value):
        """
        Set PRF value.
        """
        self.c_imagemode.prf(value)

    @property
    def rangeBandwidth(self):
        """
        Get range bandwidth.
        """
        return self.c_imagemode.rangeBandwidth()

    @rangeBandwidth.setter
    def rangeBandwidth(self, double value):
        """
        Set range bandwidth.
        """
        self.c_imagemode.rangeBandwidth(value)

    @property
    def wavelength(self):
        """
        Get radar instrument wavelength.
        """
        return self.c_imagemode.wavelength()

    @wavelength.setter
    def wavelength(self, double value):
        """
        Set radar wavelength.
        """
        self.c_imagemode.wavelength(value)

    @property
    def startingRange(self):
        """
        Get startingRange.
        """
        return self.c_imagemode.startingRange()

    @startingRange.setter
    def startingRange(self, double value):
        """
        Set startingRange.
        """
        self.c_imagemode.startingRange(value)

    @property
    def rangePixelSpacing(self):
        """
        Get rangePixelSpacing.
        """
        return self.c_imagemode.rangePixelSpacing()

    @rangePixelSpacing.setter
    def rangePixelSpacing(self, double value):
        """
        Set rangePixelSpacing.
        """
        self.c_imagemode.rangePixelSpacing(value)

    @property
    def azimuthPixelSpacing(self):
        """
        Get azimuthPixelSpacing.
        """
        return self.c_imagemode.azimuthPixelSpacing()

    @azimuthPixelSpacing.setter
    def azimuthPixelSpacing(self, double value):
        """
        Set azimuthPixelSpacing.
        """
        self.c_imagemode.azimuthPixelSpacing(value)

    @property
    def numberAzimuthLooks(self):
        """
        Get numberAzimuthLooks.
        """
        return self.c_imagemode.numberAzimuthLooks()

    @numberAzimuthLooks.setter
    def numberAzimuthLooks(self, int value):
        """
        Set numberAzimuthLooks.
        """
        self.c_imagemode.numberAzimuthLooks(value)

    @property
    def numberRangeLooks(self):
        """
        Get numberRangeLooks.
        """
        return self.c_imagemode.numberRangeLooks()

    @numberRangeLooks.setter
    def numberRangeLooks(self, int value):
        """
        Set numberRangeLooks.
        """
        self.c_imagemode.numberRangeLooks(value)

    @property
    def startAzTime(self):
        """
        Get copy of starting azimuth time.
        """
        dtime = pyDateTime.cbind(self.c_imagemode.startAzTime())
        return dtime

    @startAzTime.setter
    def startAzTime(self, pyDateTime dtime):
        """
        Set startAzTime.
        """
        self.c_imagemode.startAzTime(deref(dtime.c_datetime))

    @property
    def endAzTime(self):
        """
        Get endAzTime.
        """
        dtime = pyDateTime.cbind(self.c_imagemode.endAzTime())
        return dtime

    @endAzTime.setter
    def endAzTime(self, pyDateTime dtime):
        """
        Set endAzTime.
        """
        self.c_imagemode.endAzTime(deref(dtime.c_datetime))

# end of file
