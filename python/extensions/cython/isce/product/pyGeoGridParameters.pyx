#cython: language_level=3
#
#
#

from GeoGridParameters cimport GeoGridParameters

cdef class pyGeoGridParameters:
    """
    Cython wrapper for isce::product::GeoGridParameters.
    """
    cdef GeoGridParameters * c_geogrid
    cdef bool __owner

   def __cinit__(self):
        self.c_geogrid = new GeoGridParameters()
        self.__owner = True
        return

    def __dealloc__(self):
        if self.__owner:
            del self.c_geogrid

    @staticmethod
    def bin(pyGeoGridParameters geogrid):
        new_grid = pyGeoGridParameters()
        del new_grid.c_geogrid
        new_grid.c_geogrid = geogrid.c_geogrid
        new_grid.__owner = False
        return new_grid

    @property
    def startX(self):
        """
        Returns the starting X coordinate of the geo-grid
        """
        return self.c_geogrid.startX()

    @startX.setter
    def startX(self, double val):
        self.c_geogrid.startX(val)


    @property
    def startY(self):
        """
        Returns the starting Y coordinate of the geo-grid
        """
        return self.c_geogrid.startY()

    @startY.setter
    def startY(self, double val):
        self.c_geogrid.startY(val)

    @property
    def spacingX(self):
        """
        Returns the spacing X coordinate of the geo-grid
        """
        return self.c_geogrid.spacingX()

    @spacingX.setter
    def spacingX(self, double val):
        self.c_geogrid.spacingX(val)

    @property
    def spacingY(self):
        """
        Returns the spacing Y coordinate of the geo-grid
        """
        return self.c_geogrid.spacingY()

    @spacingY.setter
    def spacingY(self, double val):
        self.c_geogrid.spacingY(val)

    @property
    def width(self):
        return self.c_geogrid.width()

    @width.setter
    def width(self, int val):
        self.c_geogrid.width(val)

    @property
    def length(self):
        return self.c_geogrid.length()

    @length.setter
    def length(self, int val):
        self.c_geogrid.length(val)

    @property
    def epsg(self):
        return self.c_geogrid.epsg()

    @epsg.setter
    def epsg(self, int val):
        self.c_geogrid.epsg(val) 

