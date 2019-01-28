#include <gtest/gtest.h>
#include "isce/core/Constants.h"
#include "isce/core/Serialization.h"
#include "isce/io/IH5.h"
#include "isce/io/Raster.h"
#include "isce/product/Product.h"
#include "isce/geometry/Serialization.h"
#include "isce/geometry/RTC.h"

TEST(TestRTC, RunRTC) {
    // Open HDF5 file and load products
    isce::io::IH5File file("../../data/envisat.h5");
    isce::product::Product product(file);

    // Open DEM raster
    isce::io::Raster dem("../../data/srtm_cropped.tif");
    isce::product::ImageMode mode = product.complexImagery().primaryMode();
    isce::io::Raster out_raster("./rtc.bin", mode.width(), mode.length(), 1, GDT_Float32, "ENVI");

    isce::geometry::facetRTC(product, dem, out_raster);
}

TEST(TestRTC, CheckResults) {

    // Open computed integrated-area raster
    isce::io::Raster testRaster("./rtc.bin");

    // Open reference raster
    isce::io::Raster refRaster("../../data/rtc/rtc.vrt");

    ASSERT_TRUE(testRaster.width()  == refRaster.width() and
                testRaster.length() == refRaster.length());

    double error = 0; // pixelwise difference
    int nskip = 0; // number of ignored pixels

    // Valarray to hold line of data
    std::valarray<double> test(testRaster.width()), ref(refRaster.width());
    for (size_t i = 0; i < refRaster.length(); i++) {
        // Get line of data
        testRaster.getLine(test, i, 1);
        refRaster .getLine(ref,  i, 1);
        // Check each value in the line
        for (size_t j = 0; j < refRaster.width(); j++) {
            if (std::isnan(test[j]) or std::isnan(ref[j]) or
                ref[j] < 0 or test[j] < 0) {
                nskip++;
                continue;
            }
            error += std::abs(test[j] - ref[j]);
            if (std::abs(test[j] - ref[j]) > 1e-2)
                printf("%d, %d => %g ( |%g - %g| )\n", j, i, std::abs(test[j] - ref[j]), test[j], ref[j]);
        }
    }
    // Compute average over entire image
    error /= refRaster.width() * refRaster.length();

    printf("error = %g\n", error);
    printf("nskip = %d\n", nskip);

    // Enforce bound on average pixel-error
    ASSERT_TRUE(error < 1e-3);
    // Enforce bound on number of ignored pixels
    ASSERT_TRUE(nskip < 1e-3 * refRaster.width() * refRaster.length());
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
