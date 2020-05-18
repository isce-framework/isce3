#include <isce/core/forward.h>
#include <isce/io/Raster.h>
#include <isce/geometry/DEMInterpolator.h>
#include <isce/core/Projections.h>
#include <isce/product/GeoGridParameters.h>

namespace isce { namespace geocode {

    /** load DEM for a block of data
     * @param[in] demRaster a raster of the DEM
     * @param[in] demInterp DEM interpolator object
     * @param[in] proj projection object
     * @param[in] geoGrid  geo grid parameters
     * @param[in] lineStart start line 
     * @param[in] blockLength length of the block
     * @param[in] blockWidth  width of the block
     * @param[in] demMargin  extra margin
     */
    void loadDEM(isce::io::Raster demRaster,
            isce::geometry::DEMInterpolator & demInterp,
            isce::core::ProjectionBase * proj,
            const isce::product::GeoGridParameters & geoGrid,
            int lineStart, int blockLength,
            int blockWidth, double demMargin); 
}
}
