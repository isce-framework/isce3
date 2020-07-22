#cython: language_level=3
#
#
cdef extern from "isce3/product/GeoGridParameters.h" namespace "isce3::product":
    cdef cppclass GeoGridParameters:
        GeoGridParameters() except +
        GeoGridParameters(double geoGridStartX, double geoGridStartY,
                double geoGridSpacingX, double geoGridSpacingY,
                int width, int height, int epsgcode) except +

        double startX()
        void startX(double)

        double startY()
        void startY(double)

        double spacingX()
        void spacingX(double)

        double spacingY()
        void spacingY(double)

        int width()
        void width(int)

        int length()
        void length(int)

        int epsg()
        void epsg(int)

