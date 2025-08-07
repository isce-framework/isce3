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

TEST(Filter, constructAzimuthCommonbandFilter)
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

    // Get the Doppler polynomial and use it for both reference and secondary SLCs
    auto dop1 = isce3::core::avgLUT2dToLUT1d<double>(product.metadata().procInfo().dopplerCentroid('A'));
    auto dop2 = dop1;

    // get pulase repetition frequency (prf)
    double prf = swath.nominalAcquisitionPRF();
    std::cout << "prf: " << std::setprecision(16)<< prf << std::endl;

    // beta parameter for the raised cosine filter used for constructing the common azimuth band filter
    double beta = 0.25;

    // desired common azimuth band
    double commonAzimuthBandwidth = 1000.0;

    isce3::signal::Filter<float> filter;
    filter.constructAzimuthCommonbandFilter(dop1,
                                            dop2,
                                            commonAzimuthBandwidth,
                                            prf,
                                            beta,
                                            refSlc, refSpectrum,
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

    //filter.writeFilter(ncols, blockRows);

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

    //filter.writeFilter(ncols, blockRows);
    
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


