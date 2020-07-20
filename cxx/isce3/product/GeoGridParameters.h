#pragma once
#include <isce3/except/Error.h>

namespace isce { namespace product {

class GeoGridParameters {
public:
    GeoGridParameters() = default;

    inline GeoGridParameters(double geoGridStartX, double geoGridStartY,
                             double geoGridSpacingX, double geoGridSpacingY,
                             int width, int height, int epsgcode);

    void startX(double x0) { _startX = x0; }

    void startY(double y0) { _startY = y0; }

    void spacingX(double dx) { _spacingX = dx; }

    void spacingY(double dy) { _spacingY = dy; }

    void length(int l) { _length = l; };

    void width(int w) { _width = w; };

    void epsg(int e) { _epsg = e; };

    /** Get */
    double startX() const { return _startX; };

    /** Get */
    double startY() const { return _startY; };

    /** Get */
    double spacingX() const { return _spacingX; };

    /** Get */
    double spacingY() const { return _spacingY; };

    int width() const { return _width; };

    int length() const { return _length; };

    int epsg() const { return _epsg; };

protected:
    // start X position for the geocoded grid
    double _startX = 0.0;

    // start Y position for the geocoded grid
    double _startY = 0.0;

    // X spacing for the geocoded grid
    double _spacingX = 0.0;

    // Y spacing for the geocoded grid
    double _spacingY = 0.0;

    // number of pixels in east-west direction (X direction)
    int _width = 0;

    // number of lines in north-south direction (Y direction)
    int _length = 0;

    // epsg code for the output geocoded grid
    int _epsg = 4326;
};
}} // namespace isce::product

isce::product::GeoGridParameters::GeoGridParameters(
        double geoGridStartX, double geoGridStartY, double geoGridSpacingX,
        double geoGridSpacingY, int width, int length, int epsgcode)
    : // Assumption: origin is the top-left corner of the top-left pixel of the
      // grid the starting coordinate of the output geocoded grid in X
      // direction. Since the input is alwayas referring to the top-left corner
      // of the top-left pixel, we adjust to the center for internal use only
      _startX(geoGridStartX + geoGridSpacingX / 2),

      // the starting coordinate of the output geocoded grid in Y direction.
      // adjusted to the center of the pixel for internal use only
      _startY(geoGridStartY + geoGridSpacingY / 2),

      // spacing of the output geocoded grid in X
      _spacingX(geoGridSpacingX),

      // spacing of the output geocoded grid in Y
      _spacingY(geoGridSpacingY),

      // number of lines (rows) in the geocoded grid (Y direction)
      _width(width),

      // number of columns in the geocoded grid (Y direction)
      _length(length),

      // Save the EPSG code
      _epsg(epsgcode)
{
    if (geoGridSpacingY > 0.0) {
            std::string errmsg = "Y spacing can not be positive.";
            throw isce::except::OutOfRange(ISCE_SRCINFO(), errmsg);
    }
}
