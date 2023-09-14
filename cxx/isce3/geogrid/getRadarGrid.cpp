#include "getRadarGrid.h"


#include <isce3/core/Projections.h>
#include <isce3/core/LUT2d.h>
#include <isce3/io/Raster.h>
#include <isce3/geometry/geometry.h>
#include <isce3/geometry/loadDem.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/metadataCubes.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/core/DenseMatrix.h>

namespace isce3 {
namespace geogrid {

using isce3::core::Vec3;

template<class T>
static isce3::core::Matrix<T>
getNanArray(isce3::io::Raster* raster,
            const isce3::product::GeoGridParameters& geogrid)
{
    /*
    This function allocates memory for an array (`data_array`) if
    an output raster (`raster`) is provided, i.e, if `raster`
    is not a null pointer `nullptr`.
    */
    isce3::core::Matrix<T> data_array;
    if (raster != nullptr) {
        data_array.resize(geogrid.length(), geogrid.width());
    }
    data_array.fill(std::numeric_limits<T>::quiet_NaN());
    return data_array;
}

template<class T>
static void writeArray(isce3::io::Raster* raster,
        isce3::core::Matrix<T>& data_array, int band_index,
        const isce3::product::GeoGridParameters& geogrid)
{
    if (raster == nullptr) {
        return;
    }
#pragma omp critical
    {
        raster->setBlock(data_array.data(), 0, 0, data_array.width(),
                         data_array.length(), band_index + 1);
    }
    double geotransform[] = {
        geogrid.startX(),  geogrid.spacingX(), 0, geogrid.startY(), 0,
        geogrid.spacingY()};
    raster->setGeoTransform(geotransform);
    raster->setEPSG(geogrid.epsg());
}

void getRadarGrid(isce3::core::LookSide lookside,
                    const double wavelength,
                    isce3::io::Raster& dem_raster,
                    const isce3::product::GeoGridParameters& geogrid,
                    const isce3::core::Orbit& orbit,
                    const isce3::core::LUT2d<double>& native_doppler,
                    const isce3::core::LUT2d<double>& grid_doppler,
                    isce3::core::dataInterpMethod dem_interp_method,
                    const isce3::geometry::detail::Geo2RdrParams& geo2rdr_params,
                    isce3::io::Raster* interpolated_dem_raster,
                    isce3::io::Raster* slant_range_raster,
                    isce3::io::Raster* azimuth_time_raster,
                    isce3::io::Raster* incidence_angle_raster,
                    isce3::io::Raster* los_unit_vector_x_raster,
                    isce3::io::Raster* los_unit_vector_y_raster,
                    isce3::io::Raster* along_track_unit_vector_x_raster,
                    isce3::io::Raster* along_track_unit_vector_y_raster,
                    isce3::io::Raster* elevation_angle_raster,
                    isce3::io::Raster* ground_track_velocity_raster,
                    isce3::io::Raster* local_incidence_angle_raster,
                    isce3::io::Raster* projection_angle_raster,
                    isce3::io::Raster* simulated_radar_brightness_raster)
{

    pyre::journal::info_t info("isce.geogrid.getRadarGrid");

    geogrid.print();

    info << "wavelength: " << wavelength << pyre::journal::newline;
    info << "lookside: " << lookside << pyre::journal::newline;
    info << "geo2rdr threshold: " << geo2rdr_params.threshold << pyre::journal::newline;
    info << "geo2rdr max. number of iterations: " << geo2rdr_params.maxiter << pyre::journal::newline;
    info << "geo2rdr delta range: " << geo2rdr_params.delta_range << pyre::journal::endl;

    auto proj = isce3::core::makeProjection(geogrid.epsg());

    // Get DEM interpolator
    const double minX = geogrid.startX();
    const double maxX = geogrid.startX() + geogrid.spacingX() * geogrid.width();
    double minY = geogrid.startY();
    double maxY = geogrid.startY() + geogrid.spacingY() * geogrid.length();

    const double refheight = 0;
    isce3::geometry::DEMInterpolator dem_interp(refheight, dem_interp_method);
    auto error_code = loadDemFromProj(dem_raster, minX, maxX, minY, maxY,
                                      &dem_interp, proj.get());

    if (error_code != isce3::error::ErrorCode::Success) {
        std::string error_message =
                "ERROR loading DEM for given area";
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message); 
    }

    /* Get function GetDemCoords to convert DEM coordinates to the geogrid EPSG 
    coordinates */

    std::function<isce3::core::Vec3(double, double,
                       const isce3::geometry::DEMInterpolator&,
                       isce3::core::ProjectionBase*)> GetDemCoords;

    if (geogrid.epsg() == dem_raster.getEPSG()) {
        GetDemCoords = isce3::geometry::getDemCoordsSameEpsg;

    } else {
        GetDemCoords = isce3::geometry::getDemCoordsDiffEpsg;
    }

    const isce3::core::Ellipsoid& ellipsoid = proj->ellipsoid();

    // initialize output arrays
    auto interpolated_dem_array =
            getNanArray<float>(interpolated_dem_raster, geogrid);
    auto slant_range_array =
            getNanArray<double>(slant_range_raster, geogrid);
    auto azimuth_time_array =
            getNanArray<double>(azimuth_time_raster, geogrid);
    auto incidence_angle_array =
            getNanArray<float>(incidence_angle_raster, geogrid);
    auto los_unit_vector_x_array =
            getNanArray<float>(los_unit_vector_x_raster, geogrid);
    auto los_unit_vector_y_array =
            getNanArray<float>(los_unit_vector_y_raster, geogrid);
    auto along_track_unit_vector_x_array =
            getNanArray<float>(along_track_unit_vector_x_raster, geogrid);
    auto along_track_unit_vector_y_array =
            getNanArray<float>(along_track_unit_vector_y_raster, geogrid);
    auto elevation_angle_array =
            getNanArray<float>(elevation_angle_raster, geogrid);
    auto ground_track_velocity_array =
            getNanArray<double>(ground_track_velocity_raster, geogrid);
    auto local_incidence_angle_array =
            getNanArray<float>(local_incidence_angle_raster, geogrid);
    auto projection_angle_array =
            getNanArray<float>(projection_angle_raster, geogrid);
    auto simulated_radar_brightness_array =
            getNanArray<float>(simulated_radar_brightness_raster, geogrid);

#pragma omp parallel for
    for (int i = 0; i < geogrid.length(); ++i) {
        double pos_y = geogrid.startY() + (0.5 + i) * geogrid.spacingY();

        double azimuth_time = std::numeric_limits<double>::quiet_NaN();
        double native_azimuth_time = std::numeric_limits<double>::quiet_NaN();
        double slant_range = std::numeric_limits<double>::quiet_NaN();
        double native_slant_range = std::numeric_limits<double>::quiet_NaN();

        for (int j = 0; j < geogrid.width(); ++j) {
            double pos_x =
                    geogrid.startX() + (0.5 + j) * geogrid.spacingX();

            const isce3::core::Vec3 input_dem =
                    GetDemCoords(pos_x, pos_y, dem_interp, proj.get());

            if (interpolated_dem_raster != nullptr) {
                interpolated_dem_array(i, j) = input_dem[2];
            }

            // If nothing else to save, skip
            if (slant_range_raster == nullptr &&
                azimuth_time_raster == nullptr &&
                incidence_angle_raster == nullptr &&
                los_unit_vector_x_raster == nullptr &&
                los_unit_vector_y_raster == nullptr &&
                along_track_unit_vector_x_raster == nullptr &&
                along_track_unit_vector_y_raster == nullptr &&
                elevation_angle_raster == nullptr &&
                ground_track_velocity_raster == nullptr &&
                local_incidence_angle_raster == nullptr &&
                projection_angle_raster == nullptr &&
                    simulated_radar_brightness_raster == nullptr) {
                continue;
            }

            // Get target coordinates in the output projection system
            const isce3::core::Vec3 target_proj {pos_x, pos_y, input_dem[2]};

            // Get target coordinates in llh
            const isce3::core::Vec3 target_llh = proj->inverse(target_proj);

            // Get grid Doppler azimuth and slant-range position
            int converged = isce3::geometry::geo2rdr(
                    target_llh, ellipsoid, orbit, grid_doppler,
                    azimuth_time, slant_range, wavelength,
                    lookside, geo2rdr_params.threshold,
                    geo2rdr_params.maxiter, geo2rdr_params.delta_range);

            // Check convergence
            if (!converged) {
                azimuth_time = std::numeric_limits<double>::quiet_NaN();
                slant_range = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            // save grid Doppler slant-range position
            if (slant_range_raster != nullptr) {
                slant_range_array(i, j) = slant_range;
            }

            // Save grid Doppler azimuth position
            if (azimuth_time_raster != nullptr) {
                azimuth_time_array(i, j) = azimuth_time;
            }

            // If nothing else to save, skip
            if (incidence_angle_raster == nullptr &&
                los_unit_vector_x_raster == nullptr &&
                los_unit_vector_y_raster == nullptr &&
                along_track_unit_vector_x_raster == nullptr &&
                along_track_unit_vector_y_raster == nullptr &&
                elevation_angle_raster == nullptr &&
                ground_track_velocity_raster == nullptr &&
                local_incidence_angle_raster == nullptr &&
                projection_angle_raster == nullptr &&
                    simulated_radar_brightness_raster == nullptr) {
                continue;
            }

            /*
            To retrieve platform position (considering
            native Doppler), estimate native_azimuth_time
            */
            converged = isce3::geometry::geo2rdr(
                    target_llh, ellipsoid, orbit, native_doppler,
                    native_azimuth_time, native_slant_range,
                    wavelength, lookside,
                    geo2rdr_params.threshold, geo2rdr_params.maxiter,
                    geo2rdr_params.delta_range);

            // Check convergence
            if (!converged) {
                native_azimuth_time = std::numeric_limits<double>::quiet_NaN();
                native_slant_range = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            Vec3 terrain_normal_unit_vec_enu {0, 0, 1.};

            if (local_incidence_angle_raster != nullptr ||
                projection_angle_raster != nullptr ||
                    simulated_radar_brightness_raster != nullptr) {
                
                /**
                 * The terrain normal vector is calculated as:
                 * N = [-dh/dx, -dh/dy, 1] where dh/dx is the west-east slope
                 * and dh/dy is the south-north slope wrt to the geogrid. 
                 * dh, dx, and dy are computed in meters using ENU
                 * coordinates (xyz).
                 */

                float dx, dy;

                const Vec3 target_xyz = ellipsoid.lonLatToXyz(target_llh);

                // Use geogrid pixel spacing to compute dx and dy in meters
                if (geogrid.epsg() > 32600 && geogrid.epsg() < 32800) {

                    // If UTM, use geogrid spacing directly
                    dx = geogrid.spacingX();
                    dy = geogrid.spacingY();

                } else {

                    // Otherwise, convert geogrid spacing to meters
                    const isce3::core::Vec3 target_p_dx_proj
                        {pos_x + geogrid.spacingX(), pos_y, input_dem[2]};
                    const isce3::core::Vec3 target_p_dy_proj
                        {pos_x, pos_y + geogrid.spacingY(), input_dem[2]};

                    // Get target +dx and +dy coordinates in llh
                    const isce3::core::Vec3 target_p_dx_llh =
                        proj->inverse(target_p_dx_proj);
                    const isce3::core::Vec3 target_p_dy_llh =
                        proj->inverse(target_p_dy_proj);

                    // Compute difference norm in ECEF (XYZ)
                    const Vec3 target_p_dx_xyz =
                        ellipsoid.lonLatToXyz(target_p_dx_llh);
                    const Vec3 target_p_dy_xyz =
                        ellipsoid.lonLatToXyz(target_p_dy_llh);
                    dx = (target_p_dx_xyz - target_xyz).norm();
                    dy = (target_p_dy_xyz - target_xyz).norm();

                }

                // compute +/- dx and +/- dy (ENU)
                const Vec3 p_dx_enu = {+dx/2, 0, 0};
                const Vec3 m_dx_enu = {-dx/2, 0, 0};
                const Vec3 p_dy_enu = {0, +dy/2, 0};
                const Vec3 m_dy_enu = {0, -dy/2, 0};

                // Get ECEF to ENU transformation matrix
                const isce3::core::Mat3 enu2xyz =
                    isce3::core::Mat3::enuToXyz(target_llh[1], target_llh[0]);
 
                // compute +/- dx and +/- dy (ECEF)
                const Vec3 p_dx_xyz = target_xyz + enu2xyz.dot(p_dx_enu);
                const Vec3 m_dx_xyz = target_xyz + enu2xyz.dot(m_dx_enu);
                const Vec3 p_dy_xyz = target_xyz + enu2xyz.dot(p_dy_enu);
                const Vec3 m_dy_xyz = target_xyz + enu2xyz.dot(m_dy_enu);

                // compute +/- dx and +/- dy (LLH)
                Vec3 p_dx_llh, m_dx_llh;
                Vec3 p_dy_llh, m_dy_llh;
                ellipsoid.xyzToLonLat(p_dx_xyz, p_dx_llh);
                ellipsoid.xyzToLonLat(m_dx_xyz, m_dx_llh);
                ellipsoid.xyzToLonLat(p_dy_xyz, p_dy_llh);
                ellipsoid.xyzToLonLat(m_dy_xyz, m_dy_llh);

                // compute +/- dx and +/- dy (heights)
                // double rad2deg = 180.0 / M_PI;

                const double p_dx_h = dem_interp.interpolateLonLat(
                    p_dx_llh[0], p_dx_llh[1]);
                const double m_dx_h = dem_interp.interpolateLonLat(
                    m_dx_llh[0], m_dx_llh[1]);
                const double p_dy_h = dem_interp.interpolateLonLat(
                    p_dy_llh[0], p_dy_llh[1]);
                const double m_dy_h = dem_interp.interpolateLonLat(
                    m_dy_llh[0], m_dy_llh[1]);

                const double dh_dx = (p_dx_h - m_dx_h)/dx;
                const double dh_dy = (p_dy_h - m_dy_h)/dy;

                terrain_normal_unit_vec_enu = {-dh_dx, -dh_dy, +1};

                // Normalize terrain normal vector
                terrain_normal_unit_vec_enu = 
                    terrain_normal_unit_vec_enu.normalized();
            }

            isce3::geometry::writeVectorDerivedCubes(
                    i, j, native_azimuth_time, target_llh,
                    orbit, ellipsoid,
                    incidence_angle_raster, incidence_angle_array, 
                    los_unit_vector_x_raster, los_unit_vector_x_array, 
                    los_unit_vector_y_raster, los_unit_vector_y_array,
                    along_track_unit_vector_x_raster,
                    along_track_unit_vector_x_array,
                    along_track_unit_vector_y_raster,
                    along_track_unit_vector_y_array, elevation_angle_raster,
                    elevation_angle_array, ground_track_velocity_raster,
                    ground_track_velocity_array, local_incidence_angle_raster,
                    local_incidence_angle_array, projection_angle_raster,
                    projection_angle_array, simulated_radar_brightness_raster,
                    simulated_radar_brightness_array, &terrain_normal_unit_vec_enu,
                    &lookside);
        }
    }

    const int band = 0;
    writeArray(interpolated_dem_raster, interpolated_dem_array, band, geogrid);
    writeArray(slant_range_raster, slant_range_array, band, geogrid);
    writeArray(azimuth_time_raster, azimuth_time_array, band, geogrid);
    writeArray(incidence_angle_raster, incidence_angle_array, band, geogrid);
    writeArray(los_unit_vector_x_raster, los_unit_vector_x_array, band, geogrid);
    writeArray(los_unit_vector_y_raster, los_unit_vector_y_array, band, geogrid);
    writeArray(along_track_unit_vector_x_raster,
               along_track_unit_vector_x_array, band, geogrid);
    writeArray(along_track_unit_vector_y_raster,
               along_track_unit_vector_y_array, band, geogrid);
    writeArray(elevation_angle_raster, elevation_angle_array, band, geogrid);
    writeArray(ground_track_velocity_raster, ground_track_velocity_array, 
               band, geogrid);
    writeArray(local_incidence_angle_raster, local_incidence_angle_array,
               band, geogrid);
    writeArray(projection_angle_raster, projection_angle_array, band, geogrid);
    writeArray(simulated_radar_brightness_raster,
               simulated_radar_brightness_array, band, geogrid);

}

}}