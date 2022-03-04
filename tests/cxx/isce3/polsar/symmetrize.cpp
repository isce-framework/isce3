#include <complex>

#include <gtest/gtest.h>

#include <isce3/core/Constants.h>
#include <isce3/math/complexOperations.h>
#include <isce3/polsar/symmetrize.h>

using namespace isce3::math::complex_operations;

TEST(PolsarSymmetrizeTest, symmetrize)
{
    using T = float;

    const auto memory_mode_set = {isce3::core::MemoryModeBlockY::SingleBlockY,
            isce3::core::MemoryModeBlockY::MultipleBlocksY};

    const int width = 10, length = 10, nbands = 1;

    isce3::core::Matrix<std::complex<T>> hv_array(length, width);
    isce3::core::Matrix<std::complex<T>> vh_array(length, width);
    isce3::core::Matrix<std::complex<T>> output_array(length, width);

    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < width; ++j) {
            hv_array(i, j) = std::complex<T>(i, j);
            vh_array(i, j) = std::complex<T>(2 * i, 2 * j);
        }
    }

    const int x0 = 0, y0 = 0, band = 1;

    isce3::io::Raster hv_raster("symmetrize_hv_raster.bin", width, length,
            nbands, GDT_CFloat32, "ENVI");

    isce3::io::Raster vh_raster("symmetrize_vh_raster.bin", width, length,
            nbands, GDT_CFloat32, "ENVI");

    isce3::io::Raster output_raster("symmetrize_output_raster.bin", width,
            length, nbands, GDT_CFloat32, "ENVI");

    hv_raster.setBlock(hv_array.data(), x0, y0, width, length, band);
    vh_raster.setBlock(vh_array.data(), x0, y0, width, length, band);

    for (auto memory_mode : memory_mode_set) {

        isce3::polsar::symmetrizeCrossPolChannels(
                hv_raster, vh_raster, output_raster, memory_mode);

        output_raster.getBlock(
                output_array.data(), x0, y0, width, length, band);

        T symmetrization_max_error = 0;

        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < width; ++j) {
                const T error = std::abs(output_array(i, j) * 2 -
                                         (hv_array(i, j) + vh_array(i, j)));
                if (error > symmetrization_max_error) {
                    symmetrization_max_error = error;
                }
            }
        }

        std::cout << "PolSAR symmetrization max. error: "
                  << symmetrization_max_error << std::endl;
        const T symmetrization_error_threshold = 1e-6;
        EXPECT_LT(symmetrization_max_error, symmetrization_error_threshold);
    }
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
