#include <iostream>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include <isce3/io/IH5.h>
#include <isce3/io/Raster.h>
#include <isce3/product/Serialization.h>
#include <isce3/product/RadarGridProduct.h>
#include <isce3/signal/Crossmul.h>
#include <isce3/signal/Filter.h>
#include <isce3/signal/Signal.h>

TEST(Filter, constructAzimuthCommonBandFilter)
{
    //This test constructs a common azimuth band filter.

    int ncols = 500;
    int blockRows = 500;
    int nfft = ncols;

    std::valarray<std::complex<float>> refSlc(ncols*blockRows);
    std::valarray<std::complex<float>> refSpectrum(nfft*blockRows);

    // Get some metadata from an existing HDF5 file
    std::string h5file(TESTDATA_DIR "envisat.h5");

    // an HDF5 object
    isce3::io::IH5File file(h5file);

    // Create a product and swath
    isce3::product::RadarGridProduct product(file);
    const isce3::product::Swath & swath = product.swath('A');

    // Get the Doppler polynomial and use it for both refernce and secondary SLCs
    std::valarray<double> dop1(nfft);
    std::valarray<double> dop2(nfft);

    // get pulase repetition frequency (prf)
    double prf = swath.nominalAcquisitionPRF();
    std::cout << "prf: " << std::setprecision(16)<< prf << std::endl;

    // beta parameter for the raised cosine filter used for constructing the common azimuth band filter
    double beta = 0.25;

    // desired common azimuth band
    double commonAzimuthBandwidth = 1000.0;

    isce3::signal::Filter<float> filter;
    filter.constructAzimuthCommonBandCosineFilter(dop1,
                                                  dop2,
                                                  commonAzimuthBandwidth,
                                                  prf,
                                                  beta,
                                                  ncols, blockRows);
    filter.writeFilter(ncols, blockRows);
}

TEST(Filter, constructBoxcarRangeBandpassFilter)
{
    //This test constructs a boxcar range band-pass filter.
    int ncols = 500;
    int blockRows = 500;

    // memory for blocks of data and its spectrum
    std::valarray<std::complex<float>> refSlc(ncols*blockRows);
    std::valarray<std::complex<float>> refSpectrum(ncols*blockRows);

    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Create a product and swath
    isce3::product::RadarGridProduct product(file);
    const isce3::product::Swath & swath = product.swath('A');

    // get the range bandwidth
    double BW = swath.processedRangeBandwidth();

    //The bands are specified by two vectors:
    //  1) a vector of center frequencies for each sub-band
    std::valarray<double> subBandCenterFrequencies{-3.0e6, 0.0, 3e6};
    //  2) a vector of bandwidth of each sub-band
    std::valarray<double> subBandBandwidths{2.0e6, 2.0e6, 2.0e6};

    std::string filterType = "boxcar";
    // Assume range sampling frequency equals bandwidth for this test
    double rangeSamplingFrequency = BW;

    isce3::signal::Filter<float> filter;
    filter.constructRangeBandpassFilter(rangeSamplingFrequency,
                                subBandCenterFrequencies,
                                subBandBandwidths,
                                refSlc,
                                refSpectrum,
                                ncols,
                                blockRows,
                                filterType);

    filter.writeFilter(ncols, blockRows);

    // change the filter type to cosine
    filterType = "cosine";
    filter.constructRangeBandpassFilter(rangeSamplingFrequency,
                                  subBandCenterFrequencies,
                                  subBandBandwidths,
                                  refSlc,
                                  refSpectrum,
                                  ncols,
                                  blockRows,
                                  filterType);

    filter.writeFilter(ncols, blockRows);
}

TEST(Filter, constructRangeCommonBandKaiserFilter)
{
    //This test constructs a kaiser common band range band-pass FIR filter.
    int fft_size = 256;
    std::valarray<std::complex<float>> kaiser;

    double subBandCenterFrequency = 0.0;
    double subBandBandwidth = 2.0e6;
    // Assume range sampling frequency equals 1.2 times bandwidth for this test
    double rangeSamplingFrequency = 1.2 * subBandBandwidth;

    isce3::signal::Filter<float> filter;
    filter.constructRangeCommonBandKaiserFilter(subBandCenterFrequency,
                                                subBandBandwidth,
                                                rangeSamplingFrequency,
                                                fft_size,
                                                1.6,
                                                kaiser);

    ASSERT_LT(std::abs(std::abs(kaiser[0]) - 0.9997906684875488), 1.0e-6);
    ASSERT_LT(std::abs(std::abs(kaiser[127]) - 0.003600120544433594), 1.0e-6);
    ASSERT_LT(std::abs(std::abs(kaiser[255]) - 0.99970543384552), 1.0e-6);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


