#include <gtest/gtest.h>
#include <isce3/io/Raster.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/loadDem.h>
#include <isce3/geogrid/relocateRaster.h>
#include <isce3/product/GeoGridParameters.h>


using isce3::geogrid::relocateRaster;

TEST(RelocateRasterTest, CompareInterpolatedDem) {

    // Raster for existing DEM in EPSG 4326 (geographic coordinates)
    isce3::io::Raster dem_raster(TESTDATA_DIR "srtm_cropped.tif");

    // Initialize geogrid
    int epsg = 32611;
    double end_y = 3831000;
    double start_y = 3875000;
    double start_x = 611000;
    double end_x = 653000;
    double step_y = -1000;
    double step_x = 1000;

    int size_y = (end_y - start_y) / step_y;
    int size_x = (end_x - start_x) / step_x;

    auto geogrid = isce3::product::GeoGridParameters(
        start_x, start_y, step_x, step_y, size_x, size_y, epsg);

    // Initialize the interp method
    auto interp_method = isce3::core::dataInterpMethod::BIQUINTIC_METHOD;

    // Create output raster
    int nbands = 1;
    isce3::io::Raster interpolated_dem_raster(
        "relocated_faster.bin", size_x, size_y, nbands, GDT_Float32, "ENVI");

    // Call function relocateRaster()
    relocateRaster(dem_raster, geogrid, interpolated_dem_raster, 
                   interp_method);

    // Load interpolated DEM
    isce3::core::Matrix<double> interpolated_dem_array(size_y, size_x);
    interpolated_dem_raster.getBlock(interpolated_dem_array.data(), 0, 0, 
                                     size_x, size_y, 1);

    // Set DEM and geogrid projections
    auto dem_proj = isce3::core::makeProjection(dem_raster.getEPSG());
    auto geogrid_proj = isce3::core::makeProjection(epsg);

    // Get DEMInterpolator
    isce3::geometry::DEMInterpolator dem_interp(0, interp_method);
    const int margin_x_in_pixels = 100;
    const int margin_y_in_pixels = 100;
    auto error_code = loadDemFromProj(
        dem_raster, start_x, end_x, end_y, start_y, &dem_interp, 
        geogrid_proj.get(), margin_x_in_pixels, margin_y_in_pixels);
    if (error_code != isce3::error::ErrorCode::Success) {
        std::string error_message =
                "ERROR loading DEM for given area";
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_message); 
    }

    double reduction_max_error = 0;
#pragma omp parallel for reduction(max: reduction_max_error)
    for (int ii = 0; ii < size_y; ++ii) {
        for (int jj = 0; jj < size_x; ++jj) {

            double pos_y = start_y + (0.5 + ii) * step_y;
            double pos_x = start_x + (0.5 + jj) * step_x;

            auto coords_llh = geogrid_proj->inverse({pos_x, pos_y, 0});
            Vec3 dem_vect;
            dem_proj->forward(coords_llh, dem_vect);
            auto interpolated_value = dem_interp.interpolateXY(
                dem_vect[0], dem_vect[1]);

            reduction_max_error = std::max(
                std::abs(interpolated_dem_array(ii, jj) - interpolated_value),
                reduction_max_error);
        }   
    }

    ASSERT_LT(reduction_max_error, 1e-8);

}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
