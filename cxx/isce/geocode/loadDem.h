#include <isce/core/forward.h>
#include <isce/io/Raster.h>
#include <isce/geometry/DEMInterpolator.h>
#include <isce/core/Projections.h>
#include <isce/product/GeoGridParameters.h>

namespace isce { namespace geocode {

    void loadDEM(isce::io::Raster demRaster,
            isce::geometry::DEMInterpolator & demInterp,
            isce::core::ProjectionBase * proj,
            const isce::product::GeoGridParameters & geoGrid,
            int lineStart, int blockLength,
            int blockWidth, double demMargin); 
}
}
