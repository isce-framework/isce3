#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Radar cimport Radar

cdef class pyRadar:
    """
    Cython wrapper for isce::radar::Radar.

    Args:
        skew (Optional[pyLUT1d]):              pyLUT1d for skew Doppler.
        content (Optional[pyLUT1d]):           pyLUT1d for content Doppler.

    Return:
        None
    """
    # C++ class pointers
    cdef Radar * c_radar
    cdef bool __owner

    def __cinit__(self, pyLUT1d skew=None, pyLUT1d content=None):
        """
        Pre-constructor that creates a C++ isce::radar::Radar object and binds it to 
        python instance.
        """
        # Call the right constructor
        if skew is not None and content is not None:
            self.c_radar = new Radar(deref(skew.c_lut), deref(content.c_lut))
        else: 
            self.c_radar = new Radar()
        self.__owner = True

    def __dealloc__(self):
        if self.__owner:
            del self.c_radar

    @staticmethod
    cdef cbind(Radar radar):
        """
        Create a pyRadar instance with data copied from a C++ Radar object.
        """
        new_radar = pyRadar()
        del new_radar.c_radar
        new_radar.c_radar = new Radar(radar)
        new_radar.__owner = True
        return new_radar

    @property
    def contentDoppler(self):
        """
        Get a copy of the LUT1d associated with the content Doppler.
        
        Args:
            None

        Return:
            lut (pyLUT1d):                    pyLUT1d for content Doppler.
        """
        lut = pyLUT1d.cbind(self.c_radar.contentDoppler())
        return lut

    @contentDoppler.setter
    def contentDoppler(self, pyLUT1d dopp):
        """
        Set the content Doppler from LUT1d.

        Args:
            dopp (pyLUT1d):                    pyLUT1d for content Doppler.

        Return:
            None
        """
        self.c_radar.contentDoppler(deref(dopp.c_lut))

    @property
    def skewDoppler(self):
        """
        Get the LUT1d associated with the skew Doppler.

        Args:
            None

        Return:
            lut (pyLUT1d):                    pyLUT1d for skew Doppler.
        """
        lut = pyLUT1d.cbind(self.c_radar.skewDoppler())
        return lut

    @skewDoppler.setter
    def skewDoppler(self, pyLUT1d dopp):
        """
        Set the skew Doppler from LUT1d.

        Args:
            dopp (pyLUT1d):                    pyLUT1d for skew Doppler.

        Return:
            None
        """
        self.c_radar.skewDoppler(deref(dopp.c_lut))

# end of file
