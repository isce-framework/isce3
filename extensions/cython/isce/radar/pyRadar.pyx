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
        skew (Optional[pyPoly2d]):              pyPoly2d for skew Doppler.
        content (Optional[pyPoly2d]):           pyPoly2d for content Doppler.

    Return:
        None
    """
    # C++ class pointers
    cdef Radar * c_radar
    cdef bool __owner

    def __cinit__(self, pyPoly2d skew=None, pyPoly2d content=None):
        """
        Pre-constructor that creates a C++ isce::radar::Radar object and binds it to 
        python instance.
        """
        # Call the right constructor
        if skew is not None and content is not None:
            self.c_radar = new Radar(deref(skew.c_poly2d), deref(content.c_poly2d))
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
        Get a copy of the Poly2d associated with the content Doppler.
        
        Args:
            None

        Return:
            poly (pyPoly2d):                    pyPoly2d for content Doppler.
        """
        poly = pyPoly2d.cbind(self.c_radar.contentDoppler())
        return poly

    @contentDoppler.setter
    def contentDoppler(self, pyPoly2d dopp):
        """
        Set the content Doppler from Poly2d.

        Args:
            dopp (pyPoly2d):                    pyPoly2d for content Doppler.

        Return:
            None
        """
        self.c_radar.contentDoppler(deref(dopp.c_poly2d))

    @property
    def skewDoppler(self):
        """
        Get the Poly2d associated with the skew Doppler.

        Args:
            None

        Return:
            poly (pyPoly2d):                    pyPoly2d for skew Doppler.
        """
        poly = pyPoly2d.cbind(self.c_radar.skewDoppler())
        return poly

    @skewDoppler.setter
    def skewDoppler(self, pyPoly2d dopp):
        """
        Set the skew Doppler from Poly2d.

        Args:
            dopp (pyPoly2d):                    pyPoly2d for skew Doppler.

        Return:
            None
        """
        self.c_radar.skewDoppler(deref(dopp.c_poly2d))

# end of file
