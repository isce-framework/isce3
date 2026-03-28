
#include <iostream>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include <isce3/core/Utilities.h>

#include "isce3/signal/Signal.h"
#include "isce3/io/Raster.h"
#include "isce3/signal/Crossmul.h"
#include <isce3/io/IH5.h>
#include <isce3/product/RadarGridProduct.h>
#include <isce3/product/Serialization.h>

using isce3::core::avgLUT2dToLUT1d;

TEST(Crossmul, RunCrossmul)
{
    //This test creates an interferogram between an SLC and itself and checks if the
    //interferometric phase is zero.

    //a raster object for the reference SLC
    isce3::io::Raster referenceSlc(TESTDATA_DIR "warped_envisat.slc.vrt");

    //a raster object for the range and azimuth offsets
    isce3::io::Raster aziOffsets(TESTDATA_DIR "envisat_offsets.tif");
    isce3::io::Raster rngOffsets(TESTDATA_DIR "envisat_offsets.tif");

    // get the length and width of the SLC
    int width = referenceSlc.width();
    int length = referenceSlc.length();

    // a raster object for the interferogram
    isce3::io::Raster interferogram("igram.int", width, length, 1, GDT_CFloat32, "ISCE");

    std::string vsimem_ref = "/vsimem/" + getTempString("crossmul_coh");

    isce3::io::Raster coherence(vsimem_ref, width, length, 1, GDT_Float32,
                                "ENVI");

    // HDF5 file with required metadata
    std::string h5file(TESTDATA_DIR "envisat.h5");

    //H5 object
    isce3::io::IH5File file(h5file);

    // Create a product
    isce3::product::RadarGridProduct product(file);

    // get the Doppler for refernce SLC
    const char freq = 'A';
    isce3::core::LUT2d<double> dop1 = product.metadata().procInfo().dopplerCentroid(freq);
    auto swath = product.swath(freq);

    // Since this test careates an interferogram between the reference SLC and itself,
    // the second Doppler is the same as the first
    isce3::core::LUT2d<double> dop2 = dop1;

    //instantiate the Crossmul class
    isce3::signal::Crossmul crsmul;

    // set Doppler  for refernce and secondary SLCs
    crsmul.doppler(dop1, dop2);

    // set number of interferogram looks in range
    crsmul.rangeLooks(1);

    // set number of interferogram looks in azimuth
    crsmul.azimuthLooks(1);

    // set the product information for the common band filtering and flattenning
    const double wavelength = swath.processedWavelength();
    const double azimuthBandwidth = swath.processedAzimuthBandwidth();
    const double rangeBandwidth = swath.processedRangeBandwidth();
    const double prf = swath.nominalAcquisitionPRF();
    const double rngPixelSampling =  swath.rangePixelSpacing();
    const double startAzimuthTime = swath.zeroDopplerTime()[0];
    const double startSlantRange = swath.slantRange()[0];

    crsmul.wavelength(wavelength);
    crsmul.prf(prf);
    crsmul.rangePixelSpacing(rngPixelSampling);
    crsmul.azimuthBandwidth(azimuthBandwidth);
    crsmul.rangeBandwidth(rangeBandwidth);
    crsmul.refStartRange(startSlantRange);
    crsmul.secStartRange(startSlantRange);
    crsmul.refStartAzimuthTime(startAzimuthTime);
    crsmul.secStartAzimuthTime(startAzimuthTime);

    // Enable the flattening and common band filtering
    crsmul.doFlatten(true);
    crsmul.doCommonRangeBandFilter(true);
    crsmul.doCommonAzimuthBandFilter(true);

    // running crossmul
    crsmul.crossmul(referenceSlc, referenceSlc,
                    interferogram, coherence,
                    &rngOffsets, &aziOffsets);

    // an array for the computed interferogram
    std::valarray<std::complex<float>> data(width*length);

    // get a block of the computed interferogram
    interferogram.getBlock(data, 0, 0, width, length);

    // check if the interferometric phase is zero
    double err = 0.0;
    double max_err = 0.0;
    for ( size_t i = 0; i < data.size(); ++i ) {
          err = std::arg(data[i]);
          if (std::abs(err) > max_err){
              max_err = std::abs(err);
          }
      }

      ASSERT_LT(max_err, 1.0e-6);
}

TEST(Crossmul, RunCrossmulMLook)
{
    //This test creates an interferogram between an SLC and itself and checks if the
    //interferometric phase is zero.

    //a raster object for the reference SLC
    isce3::io::Raster referenceSlc(TESTDATA_DIR "warped_envisat.slc.vrt");

    //a raster object for the range and azimuth offsets
    isce3::io::Raster aziOffsets(TESTDATA_DIR "envisat_offsets.tif");
    isce3::io::Raster rngOffsets(TESTDATA_DIR "envisat_offsets.tif");

    // define looks
    const int rngLooks = 3;
    const int azLooks = 13;

    // get the length and width of the SLC
    const int width = referenceSlc.width() / rngLooks;
    const int length = referenceSlc.length() / azLooks;

    // a raster object for the interferogram
    isce3::io::Raster interferogram("igram.int", width, length, 1, GDT_CFloat32, "ISCE");

    std::string vsimem_ref = "/vsimem/" + getTempString("crossmul_ml_coh");

    isce3::io::Raster coherence(vsimem_ref, width, length, 1, GDT_Float32,
                                "ENVI");

    // HDF5 file with required metadata
    std::string h5file(TESTDATA_DIR "envisat.h5");

    //H5 object
    isce3::io::IH5File file(h5file);

    // Create a product
    isce3::product::RadarGridProduct product(file);

    // get the Doppler polynomial for refernce SLC
    const char freq = 'A';
    isce3::core::LUT2d<double> dop1 = product.metadata().procInfo().dopplerCentroid(freq);
    auto swath = product.swath(freq);

    // Since this test careates an interferogram between the reference SLC and itself,
    // the second Doppler is the same as the first
    isce3::core::LUT2d<double> dop2 = dop1;

    //instantiate the Crossmul class
    isce3::signal::Crossmul crsmul;

    // set Doppler polynomials for reference and secondary SLCs
    crsmul.doppler(dop1, dop2);

    // set number of interferogram looks in range
    crsmul.rangeLooks(rngLooks);

    // set number of interferogram looks in azimuth
    crsmul.azimuthLooks(azLooks);

    // set the product information for the common band filtering and flattenning
    const double wavelength = swath.processedWavelength();
    const double azimuthBandwidth = swath.processedAzimuthBandwidth();
    const double rangeBandwidth = swath.processedRangeBandwidth();
    const double prf = swath.nominalAcquisitionPRF();
    const double rngPixelSampling =  swath.rangePixelSpacing();
    const double startAzimuthTime = swath.zeroDopplerTime()[0];
    const double startSlantRange = swath.slantRange()[0];

    crsmul.wavelength(wavelength);
    crsmul.prf(prf);
    crsmul.rangePixelSpacing(rngPixelSampling);
    crsmul.azimuthBandwidth(azimuthBandwidth);
    crsmul.rangeBandwidth(rangeBandwidth);
    crsmul.refStartRange(startSlantRange);
    crsmul.secStartRange(startSlantRange);
    crsmul.refStartAzimuthTime(startAzimuthTime);
    crsmul.secStartAzimuthTime(startAzimuthTime);

    // Enable the flattening and common band filtering
    crsmul.doFlatten(true);
    crsmul.doCommonRangeBandFilter(true);
    crsmul.doCommonAzimuthBandFilter(true);

    // running crossmul
    crsmul.crossmul(referenceSlc, referenceSlc,
                    interferogram, coherence,
                    &rngOffsets, &aziOffsets);

    // an array for the computed interferogram
    std::valarray<std::complex<float>> data(width*length);

    // get a block of the computed interferogram
    interferogram.getBlock(data, 0, 0, width, length);

    // check if the interferometric phase is zero
    double err = 0.0;
    double max_err = 0.0;
    for ( size_t i = 0; i < data.size(); ++i ) {
        err = std::arg(data[i]);
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
      }

      ASSERT_LT(max_err, 1.0e-6);
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


