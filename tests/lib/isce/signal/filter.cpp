
#include <iostream>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include "isce/signal/Signal.h"
#include "isce/io/Raster.h"
#include "isce/signal/Crossmul.h"
#include <isce/io/IH5.h>
#include <isce/radar/Radar.h>
#include <isce/radar/Serialization.h>
#include <isce/product/Serialization.h>

TEST(Filter, constructAzimuthCommonbandFilter)
{
    //This test constructs a common azimuth band filter.

    int ncols = 500;
    int blockRows = 500;
    int nfft = ncols;
    int oversample = 2;

    std::valarray<std::complex<float>> refSlc(ncols*blockRows);
    std::valarray<std::complex<float>> refSpectrum(nfft*blockRows);

    // Get some metadata from an existing HDF5 file
    //std::string h5file("/Users/fattahi/tools/ISCE3_forked/src/isce/tests/lib/isce/data/envisat.h5");
    std::string h5file("../data/envisat.h5");

    // an HDF5 object
    isce::io::IH5File file(h5file);

    // a radra object
    isce::radar::Radar instrument;

    // Open group containing instrument data
    isce::io::IGroup group = file.openGroup("/science/metadata/instrument_data");

    // Deserialize the radar instrument
    isce::radar::loadFromH5(group, instrument);

    // Open group for image mode
    isce::io::IGroup modeGroup = file.openGroup("/science/complex_imagery");

    // Get the Doppler polynomial and use it for both refernce and secondary SLCs
    isce::core::Poly2d dop1 = instrument.contentDoppler();
    isce::core::Poly2d dop2 = instrument.contentDoppler();

    // Instantiate an ImageMode object
    isce::product::ImageMode mode;
   
    // Deserialize the primary_mode
    isce::product::loadFromH5(modeGroup, mode, "aux");
 
    // get pulase repetition frequency (prf)
    double prf = mode.prf(); 
    std::cout << "prf: " << std::setprecision(16)<< prf << std::endl;

    // beta parameter for the raised cosine filter used for constructing the common azimuth band filter
    double beta = 0.25;

    // desired common azimuth band
    double commonAzimuthBandwidth = 1000.0;

    isce::signal::Filter<float> filter;
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

    std::string h5file("../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Open group for image mode
    isce::io::IGroup modeGroup = file.openGroup("/science/complex_imagery");
    
    // Instantiate an ImageMode object
    isce::product::ImageMode mode;

    // Deserialize the primary_mode
    isce::product::loadFromH5(modeGroup, mode, "aux");

    // get the range bandwidth
    double BW = mode.rangeBandwidth();
    
    //The bands are specified by two vectors:
    //  1) a vector of center frequencies for each sub-band
    std::valarray<double> subBandCenterFrequencies{-3.0e6, 0.0, 3e6}; 
    //  2) a vector of bandwidth of each sub-band
    std::valarray<double> subBandBandwidths{2.0e6, 2.0e6, 2.0e6};

    std::string filterType = "boxcar";
    // Assume range sampling frequency equals bandwidth for this test
    double rangeSamplingFrequency = BW;

    isce::signal::Filter<float> filter;
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


