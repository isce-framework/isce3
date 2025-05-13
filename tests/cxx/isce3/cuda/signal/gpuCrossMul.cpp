
#include <iostream>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <cmath>
#include <complex>

#include <gtest/gtest.h>

#include <isce3/core/Utilities.h>

#include "isce3/io/Raster.h"
#include <isce3/io/IH5.h>
#include <isce3/product/RadarGridProduct.h>
#include <isce3/product/Serialization.h>

#include "isce3/cuda/signal/gpuCrossMul.h"

using isce3::core::avgLUT2dToLUT1d;

TEST(gpuCrossmul, Crossmul)
{
    //This test creates an interferogram between an SLC and itself and checks if the
    //interferometric phase is zero.

    //a raster object for the reference SLC
    isce3::io::Raster referenceSlc(TESTDATA_DIR "warped_envisat.slc.vrt");

    // get the length and width of the SLC
    int width = referenceSlc.width();
    int length = referenceSlc.length();

    // a raster object for the interferogram
    isce3::io::Raster interferogram("igram.int", width, length, 1, GDT_CFloat32, "ISCE");

    std::string vsimem_ref = "/vsimem/" + getTempString("gpu_crossmul_coh");

    isce3::io::Raster coherence(vsimem_ref, width, length, 1, GDT_Float32,
                                "ENVI");

    // HDF5 file with required metadata
    std::string h5file(TESTDATA_DIR "envisat.h5");

    //H5 object
    isce3::io::IH5File file(h5file);

    // Create a product
    isce3::product::RadarGridProduct product(file);

    // get the Doppler polynomial for reference SLC
    const isce3::core::LUT1d<double> dop1 =
        avgLUT2dToLUT1d<double>(product.metadata().procInfo().dopplerCentroid('A'));

    // Since this test careates an interferogram between the reference SLC and itself,
    // the second Doppler is the same as the first
    isce3::core::LUT1d<double> dop2 = dop1;

    //instantiate the Crossmul class
    isce3::cuda::signal::gpuCrossmul crsmul;

    // set Doppler polynomials for reference and secondary SLCs
    crsmul.doppler(dop1, dop2);

    // set number of interferogram looks in range
    crsmul.rangeLooks(1);

    // set number of interferogram looks in azimuth
    crsmul.azimuthLooks(1);

    // running crossmul
    crsmul.crossmul(referenceSlc, referenceSlc, interferogram, coherence);

    // an array for the computed interferogram
    std::valarray<std::complex<float>> data(width*length);

    // get a block of the computed interferogram
    interferogram.getBlock(data, 0, 0, width, length);

    // check if the interferometric phase is zero
    double err = 0.0;
    double max_err = 0.0;
    for ( size_t i = 0; i < data.size(); ++i ) {
          err = std::arg(data[i]);
          if (std::abs(err) > max_err) {
              max_err = std::abs(err);
        }
    }

    ASSERT_LT(max_err, 1.0e-6);
}


TEST(gpuCrossmul, MultilookCrossmul)
{
    //This test creates an interferogram between an SLC and itself and checks if the
    //interferometric phase is zero.

    //a raster object for the reference SLC
    isce3::io::Raster referenceSlc(TESTDATA_DIR "warped_envisat.slc.vrt");

    // define looks
    const int rngLooks = 3;
    const int azLooks = 13;

    // get the length and width of the SLC
    const int width = referenceSlc.width() / rngLooks;
    const int length = referenceSlc.length() / azLooks;

    // a raster object for the interferogram
    isce3::io::Raster interferogram("igram.int", width, length, 1, GDT_CFloat32, "ISCE");

    std::string vsimem_ref = "/vsimem/" + getTempString("gpu_crossmul_ml_coh");

    isce3::io::Raster coherence(vsimem_ref, width, length, 1, GDT_Float32,
                                "ENVI");

    // HDF5 file with required metadata
    std::string h5file(TESTDATA_DIR "envisat.h5");

    //H5 object
    isce3::io::IH5File file(h5file);

    // Create a product
    isce3::product::RadarGridProduct product(file);

    // get the Doppler polynomial for reference SLC
    isce3::core::LUT1d<double> dop1 =
        avgLUT2dToLUT1d(product.metadata().procInfo().dopplerCentroid('A'));

    // Since this test careates an interferogram between the reference SLC and itself,
    // the second Doppler is the same as the first
    isce3::core::LUT1d<double> dop2 = dop1;

    //instantiate the Crossmul class
    isce3::cuda::signal::gpuCrossmul crsmul;

    // set Doppler polynomials for reference and secondary SLCs
    crsmul.doppler(dop1, dop2);

    // set number of interferogram looks in range
    crsmul.rangeLooks(rngLooks);

    // set number of interferogram looks in azimuth
    crsmul.azimuthLooks(azLooks);

    // running crossmul
    crsmul.crossmul(referenceSlc, referenceSlc, interferogram, coherence);

    // an array for the computed interferogram
    std::valarray<std::complex<float>> data(width*length);

    // get a block of the computed interferogram
    interferogram.getBlock(data, 0, 0, width, length);

    // check if the interferometric phase is zero
    double err = 0.0;
    double max_err = 0.0;
    for ( size_t i = 0; i < data.size(); ++i ) {
          err = std::arg(data[i]);
          if (std::abs(err) > max_err) {
              max_err = std::abs(err);
        }
    }

    ASSERT_LT(max_err, 1.0e-7);
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


