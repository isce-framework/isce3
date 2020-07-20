#pragma once

#include <array>

namespace isce { namespace io { namespace gdal {

/**
 * Transform coefficients for transforming from (pixel, line) coordinates in
 * raster space to (x, y) coordinates in projection space
 *
 * Only separable, axis-aligned transforms are supported. That is, the
 * transform must be able to be decomposed into \code f(pixel) -> x \endcode
 * and \code g(line) -> y \endcode
 */
struct GeoTransform {

    /** Default constructor (identity transform) */
    GeoTransform() = default;

    /**
     * Constructor
     *
     * \param[in] x0 Left edge of the left-most pixel in the raster
     * \param[in] y0 Upper edge of the upper-most line in the raster
     * \param[in] dx Pixel width
     * \param[in] dy Line height
     */
    GeoTransform(double x0, double y0, double dx, double dy) : x0(x0), y0(y0), dx(dx), dy(dy) {}

    /**
     * Construct from GDAL affine transform coefficients.
     *
     * Only separable, axis-aligned transforms are supported,
     * ( \code coeffs[2] == 0. \endcode and \code coeffs[4] == 0. \endcode )
     *
     * \throws isce::except::InvalidArgument if the supplied transform is unsupported
     *
     * \param[in] coeffs Affine transform coefficients such as those produced by GDALGetGeoTransform()
     */
    GeoTransform(const std::array<double, 6> & coeffs);

    /**
     * Get equivalent GDAL affine transform coefficients.
     *
     * \returns Affine transform coefficients
     */
    std::array<double, 6> getCoeffs() const;

    double transformX(int pixel) const { return x0 + (pixel + 0.5) * dx; }

    double transformY(int line) const { return y0 + (line + 0.5) * dy; }

    /** True if the transform is identity. */
    bool isIdentity() const { return x0 == 0. && y0 == 0. && dx == 1. && dy == 1.; }

    double x0 = 0.;
    double y0 = 0.;
    double dx = 1.;
    double dy = 1.;
};

bool operator==(const GeoTransform &, const GeoTransform &);
bool operator!=(const GeoTransform &, const GeoTransform &);

}}}
