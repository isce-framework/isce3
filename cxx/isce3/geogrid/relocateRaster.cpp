#include "relocateRaster.h"

#include <string>
#include <isce3/core/Projections.h>
#include <isce3/io/Raster.h>
#include <isce3/geometry/geometry.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/loadDem.h>
#include <isce3/product/GeoGridParameters.h>

namespace isce3 {
namespace geogrid {


template<class T>
static isce3::core::Matrix<T>
getNanArray(const isce3::product::GeoGridParameters& geogrid)
{
    isce3::core::Matrix<T> data_array(geogrid.length(), geogrid.width());
    data_array.fill(std::numeric_limits<T>::quiet_NaN());
    return data_array;
}

template<class T>
static void writeArray(isce3::io::Raster& raster,
        isce3::core::Matrix<T>& data_array, int band_index)
{
    raster.setBlock(data_array.data(), 0, 0, data_array.width(),
                    data_array.length(), band_index + 1);
}

void _validateRasters(isce3::io::Raster& input_raster,
                      const isce3::product::GeoGridParameters& geogrid,
                      isce3::io::Raster& output_raster) {

    if (input_raster.getEPSG() < 0) {
        std::string error_message =
            "ERROR invalid input raster EPSG code: " +
            std::to_string(input_raster.getEPSG());
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message); 
    }

    if (geogrid.epsg() < 0) {
        std::string error_message =
            "ERROR invalid geogrid EPSG code: " +
            std::to_string(geogrid.epsg());
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message); 
    }

    if (geogrid.length() != output_raster.length() or
            geogrid.width() != output_raster.width()) {
        std::string error_message =
            "ERROR geogrid and output rasters have different dimensions"; 
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message); 
    }

    if (input_raster.numBands() != output_raster.numBands()) {
        std::string error_message =
            "ERROR input and output rasters have different number of bands: " +
            std::to_string(input_raster.numBands()) + " (input raster) vs " +
            std::to_string(input_raster.numBands()) + " (output raster)";
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message); 
    }
}

void relocateRaster(isce3::io::Raster& input_raster,
                    const isce3::product::GeoGridParameters& geogrid,
                    isce3::io::Raster& output_raster,
                    isce3::core::dataInterpMethod interp_method) {

    pyre::journal::info_t info("isce.geogrid.relocateRaster");

    _validateRasters(input_raster, geogrid, output_raster);

    geogrid.print();

    auto proj = isce3::core::makeProjection(geogrid.epsg());
    
    double refheight = 0;
    isce3::geometry::DEMInterpolator dem_interp(refheight, interp_method);
    const double minX = geogrid.startX();
    const double maxX = geogrid.startX() + (geogrid.spacingX() * 
                                            geogrid.width());
    double minY = geogrid.startY();
    double maxY = geogrid.startY() + geogrid.spacingY() * geogrid.length();

    const int dem_margin_in_pixels = 100;

    for (int band_index = 0; band_index < input_raster.numBands();
         ++band_index) {

        auto error_code = loadDemFromProj(input_raster, minX, maxX, minY,
                                          maxY, &dem_interp, proj.get(), 
                                          dem_margin_in_pixels,
                                          dem_margin_in_pixels,
                                          band_index + 1);

        if (error_code != isce3::error::ErrorCode::Success) {
            std::string error_message =
                    "ERROR loading raster for given area";
            throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message); 
        }
        
        /*
        Get DEM interpolator and function getDemCoords to convert DEM coordinates
        to the geogrid EPSG coordinates:
        */
        std::function<isce3::core::Vec3(double, double,
                        const isce3::geometry::DEMInterpolator&,
                        isce3::core::ProjectionBase*)> getDemCoords;

        if (geogrid.epsg() == input_raster.getEPSG()) {
            getDemCoords = isce3::geometry::getDemCoordsSameEpsg;

        } else {
            getDemCoords = isce3::geometry::getDemCoordsDiffEpsg;
        }

        auto interpolated_dem_array = getNanArray<float>(geogrid);

        #pragma omp parallel for
        for (int i = 0; i < geogrid.length(); ++i) {
            double pos_y = geogrid.startY() + (0.5 + i) * geogrid.spacingY();
            for (int j = 0; j < geogrid.width(); ++j) {
                double pos_x =
                        geogrid.startX() + (0.5 + j) * geogrid.spacingX();

                const isce3::core::Vec3 input_dem =
                        getDemCoords(pos_x, pos_y, dem_interp, proj.get());

                interpolated_dem_array(i, j) = input_dem[2];
            }
        }

        writeArray(output_raster, interpolated_dem_array, band_index);

    }

    double geotransform[] = {
            geogrid.startX(),  geogrid.spacingX(), 0, geogrid.startY(), 0,
            geogrid.spacingY()};

    output_raster.setGeoTransform(geotransform);
    output_raster.setEPSG(geogrid.epsg());
    
}

}}
