#include "getGeolocationGrid.h"


#include <isce3/core/Projections.h>
#include <isce3/core/LUT2d.h>
#include <isce3/io/Raster.h>
#include <isce3/geometry/geometry.h>
#include <isce3/geometry/loadDem.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/boundingbox.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/geometry/metadataCubes.h>

namespace isce3 {
namespace geometry {


template<class T>
static isce3::core::Matrix<T>
getNanArrayRadarGrid(isce3::io::Raster* raster,
              const isce3::product::RadarGridParameters& radar_grid)
{
    /*
    This function allocates memory for an array (`data_array`) if
    an output raster (`raster`) is provided, i.e, if `raster`
    is not a null pointer `nullptr`.
    */
    isce3::core::Matrix<T> data_array;
    if (raster != nullptr) {
        data_array.resize(radar_grid.length(), radar_grid.width());
    }
    data_array.fill(std::numeric_limits<T>::quiet_NaN());
    return data_array;
}

template<class T>
static void writeArray(isce3::io::Raster* raster,
        isce3::core::Matrix<T>& data_array, int band_index)
{
    if (raster == nullptr) {
        return;
    }
#pragma omp critical
    {
        raster->setBlock(data_array.data(), 0, 0, data_array.width(),
                         data_array.length(), band_index + 1);
    }
}

void getGeolocationGrid(isce3::io::Raster& dem_raster,
                        const isce3::product::RadarGridParameters& radar_grid,
                        const isce3::core::Orbit& orbit,
                        const isce3::core::LUT2d<double>& native_doppler,
                        const isce3::core::LUT2d<double>& grid_doppler,
                        const int epsg,
                        isce3::core::dataInterpMethod dem_interp_method,
                        const isce3::geometry::detail::Rdr2GeoParams& rdr2geo_params,
                        const isce3::geometry::detail::Geo2RdrParams& geo2rdr_params,
                        isce3::io::Raster* interpolated_dem_raster,
                        isce3::io::Raster* coordinate_x_raster,
                        isce3::io::Raster* coordinate_y_raster,
                        isce3::io::Raster* incidence_angle_raster,
                        isce3::io::Raster* los_unit_vector_x_raster,
                        isce3::io::Raster* los_unit_vector_y_raster,
                        isce3::io::Raster* along_track_unit_vector_x_raster,
                        isce3::io::Raster* along_track_unit_vector_y_raster,
                        isce3::io::Raster* elevation_angle_raster,
                        isce3::io::Raster* ground_track_velocity_raster)
{

    pyre::journal::info_t info("isce.geometry.getGeolocationGrid");

    info << "epsg: " << epsg << pyre::journal::newline;
    info << "wavelength: " << radar_grid.wavelength() << pyre::journal::newline;
    info << "lookside: " << radar_grid.lookSide() << pyre::journal::newline;
    info << "rdr2geo threshold: " << rdr2geo_params.threshold << pyre::journal::newline;
    info << "rdr2geo max. number of iterations: " << rdr2geo_params.extraiter << pyre::journal::newline;
    info << "rdr2geo extra number of iterations: " << rdr2geo_params.maxiter << pyre::journal::endl;
    info << "geo2rdr threshold: " << geo2rdr_params.threshold << pyre::journal::newline;
    info << "geo2rdr max. number of iterations: " << geo2rdr_params.maxiter << pyre::journal::newline;
    info << "geo2rdr delta range: " << geo2rdr_params.delta_range << pyre::journal::endl;

    auto proj = isce3::core::makeProjection(epsg);

    const isce3::core::Ellipsoid& ellipsoid = proj->ellipsoid();

    // initialize output arrays
    auto interpolated_dem_array =
            getNanArrayRadarGrid<float>(interpolated_dem_raster, radar_grid);
    auto coordinate_x_array =
            getNanArrayRadarGrid<double>(coordinate_x_raster, radar_grid);
    auto coordinate_y_array =
            getNanArrayRadarGrid<double>(coordinate_y_raster, radar_grid);
    auto incidence_angle_array =
            getNanArrayRadarGrid<float>(incidence_angle_raster, radar_grid);
    auto los_unit_vector_x_array =
            getNanArrayRadarGrid<float>(los_unit_vector_x_raster, radar_grid);
    auto los_unit_vector_y_array =
            getNanArrayRadarGrid<float>(los_unit_vector_y_raster, radar_grid);
    auto along_track_unit_vector_x_array =
            getNanArrayRadarGrid<float>(along_track_unit_vector_x_raster, 
                radar_grid);
    auto along_track_unit_vector_y_array =
            getNanArrayRadarGrid<float>(along_track_unit_vector_y_raster, 
                radar_grid);
    auto elevation_angle_array =
            getNanArrayRadarGrid<float>(elevation_angle_raster, radar_grid);
    auto ground_track_velocity_array =
            getNanArrayRadarGrid<double>(ground_track_velocity_raster, 
                radar_grid);

    BoundingBox bbox = getGeoBoundingBoxHeightSearch(radar_grid, orbit,
                                                     proj.get(), grid_doppler);

    isce3::geometry::DEMInterpolator dem_interp(0, dem_interp_method);
    auto error_code = loadDemFromProj(dem_raster, bbox.MinX, bbox.MaxX, 
                                      bbox.MinY, bbox.MaxY,
                                      &dem_interp, proj.get());

    if (error_code != isce3::error::ErrorCode::Success) {
        std::string error_message =
                "ERROR loading DEM for given area";
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message); 
    }

    /*
    Initialize with `nullptr` variable pointers that are unused but required
    in the call to `writeVectorDerivedCubes()`
    */
    isce3::io::Raster * local_incidence_angle_raster = nullptr;
    isce3::core::Matrix<float> local_incidence_angle_array;
    isce3::io::Raster * projection_angle_raster = nullptr;
    isce3::core::Matrix<float> projection_angle_array;
    isce3::io::Raster * simulated_radar_brightness_raster = nullptr;
    isce3::core::Matrix<float> simulated_radar_brightness_array;
    isce3::core::Vec3* terrain_normal_vector = nullptr;
    isce3::core::LookSide* lookside = nullptr;

#pragma omp parallel for
    for (int i = 0; i < radar_grid.length(); ++i) {
        const int height_first_guess = 0;
        double native_azimuth_time = std::numeric_limits<double>::quiet_NaN();
        double native_slant_range = std::numeric_limits<double>::quiet_NaN();
        double az_time = radar_grid.sensingTime(i);
        for (int j = 0; j < radar_grid.width(); ++j) {
            double slant_range = radar_grid.slantRange(j);
            Vec3 target_llh;
            /*
            Skip processing for radar grid points outside grid doppler
            */
            if (!grid_doppler.contains(az_time, slant_range)) {
                continue;
            }

            /*
            Get target position (target_llh) considering grid Doppler
            */
            double fd = grid_doppler.eval(az_time, slant_range);
            target_llh[2] = height_first_guess;
            auto converged =
                    rdr2geo(az_time, slant_range, fd, orbit, ellipsoid,
                            dem_interp, target_llh, radar_grid.wavelength(),
                            radar_grid.lookSide(), rdr2geo_params.threshold,
                            rdr2geo_params.maxiter, rdr2geo_params.extraiter);

            // Check convergence
            if (!converged) {
                continue;
            }

            // Get target position in the output proj system
            isce3::core::Vec3 target_proj = proj->forward(target_llh);
            if (coordinate_x_raster != nullptr) {
                coordinate_x_array(i, j) = target_proj[0];
            }
            if (coordinate_y_raster != nullptr) {
                coordinate_y_array(i, j) = target_proj[1];
            }

            if (interpolated_dem_raster != nullptr) {
                interpolated_dem_array(i, j) = target_llh[2];
            }

            // If nothing else to save, skip
            if (incidence_angle_raster == nullptr &&
                los_unit_vector_x_raster == nullptr &&
                los_unit_vector_y_raster == nullptr &&
                along_track_unit_vector_x_raster == nullptr &&
                along_track_unit_vector_y_raster == nullptr &&
                elevation_angle_raster == nullptr &&
                ground_track_velocity_raster == nullptr) {
                continue;
            }

            /*
            To retrieve platform position (considering
            native Doppler), estimate native_azimuth_time 
            */
            if (std::isnan(native_azimuth_time)) {
                native_azimuth_time = az_time;
            }
            if (std::isnan(native_slant_range)) {
                native_slant_range = slant_range;
            }

            converged = geo2rdr(target_llh, ellipsoid, orbit, native_doppler,
                    native_azimuth_time, native_slant_range,
                    radar_grid.wavelength(), radar_grid.lookSide(),
                    geo2rdr_params.threshold, geo2rdr_params.maxiter,
                    geo2rdr_params.delta_range);

            // Check convergence
            if (!converged) {
                /*
                If didn't converge, use `az_time` and `slant_range`
                from zero-Doppler as an initial solution for next
                iteration
                */
                native_azimuth_time = az_time;
                native_slant_range = slant_range;
                continue;
            }

            writeVectorDerivedCubes(i, j, native_azimuth_time, target_llh,
                    orbit, ellipsoid,
                    incidence_angle_raster, incidence_angle_array,
                    los_unit_vector_x_raster, los_unit_vector_x_array,
                    los_unit_vector_y_raster, los_unit_vector_y_array,
                    along_track_unit_vector_x_raster,
                    along_track_unit_vector_x_array,
                    along_track_unit_vector_y_raster,
                    along_track_unit_vector_y_array, 
                    elevation_angle_raster,
                    elevation_angle_array,
                    ground_track_velocity_raster,
                    ground_track_velocity_array, local_incidence_angle_raster,
                    local_incidence_angle_array, projection_angle_raster,
                    projection_angle_array, simulated_radar_brightness_raster,
                    simulated_radar_brightness_array, terrain_normal_vector,
                    lookside);
        }
    }

    const int band = 0;
    writeArray(interpolated_dem_raster, interpolated_dem_array, band);
    writeArray(coordinate_x_raster, coordinate_x_array, band);
    writeArray(coordinate_y_raster, coordinate_y_array, band);
    writeArray(incidence_angle_raster, incidence_angle_array, band);
    writeArray(los_unit_vector_x_raster, los_unit_vector_x_array, band);
    writeArray(los_unit_vector_y_raster, los_unit_vector_y_array, band);
    writeArray(along_track_unit_vector_x_raster,
                along_track_unit_vector_x_array, band);
    writeArray(along_track_unit_vector_y_raster,
                along_track_unit_vector_y_array, band);
    writeArray(elevation_angle_raster, elevation_angle_array, band);
    writeArray(ground_track_velocity_raster, ground_track_velocity_array, 
               band);

}

}}