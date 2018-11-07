
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

//TEST(Crossmul, InterferogramZero)
//{
int main(){
    //This test creates an interferogram between an SLC and itself and checks if the phase is zero.
    //
    isce::io::Raster referenceSlc("/Users/fattahi/tools/ISCE3_forked/src/isce/tests/lib/isce/data/warped_envisat.slc.vrt");

    
    // make a raster of reference SLC
    //isce::io::Raster refSlc("../data/warped_envisat.slc.vrt");
    
    int ncols = 500;
    int blockRows = 500;
    int nfft = ncols;
    int oversample = 2;

    std::valarray<std::complex<float>> refSlc(ncols*blockRows);
    std::valarray<std::complex<float>> refSpectrum(nfft*blockRows);

    // Open the file
    std::string h5file("/Users/fattahi/tools/ISCE3_forked/src/isce/tests/lib/isce/data/envisat.h5");
    isce::io::IH5File file(h5file);
    isce::radar::Radar instrument;
    isce::radar::load(file, instrument);
    isce::core::Poly2d dop1 = instrument.contentDoppler();
    isce::core::Poly2d dop2 = instrument.contentDoppler();

    // Instantiate an ImageMode object
    isce::product::ImageMode mode;
    isce::product::load(file, mode, "aux");


    double prf = mode.prf(); 
    //1652.415691672402;
    double beta = 0.25;
    double commonAzimuthBandwidth = 2000;

    isce::signal::Filter<float> filter;
    filter.constructAzimuthCommonbandFilter(dop1,
                                            dop2,
                                            commonAzimuthBandwidth,
                                            prf,
                                            beta,
                                            refSlc, refSpectrum,
                                            ncols, blockRows);
    filter.writeFilter(ncols, blockRows);
    
    double BW = mode.rangeBandwidth();
    
    //Assume Range sampling Frequency to be the same as
    std::valarray<double> subBandCenterFrequencies{-3.0e6, 0.0, 3e6}; 
    std::valarray<double> subBandBandwidths{2.0e6, 2.0e6, 2.0e6};

    //std::valarray<double> subBandCenterFrequencies{0.0};
    //std::valarray<double> subBandBandwidths{8.0e6,};
    std::string filterType = "boxcar";

    filter.constructRangeBandpassFilter(BW,
                                subBandCenterFrequencies,
                                subBandBandwidths,
                                refSlc,
                                refSpectrum,
                                ncols,
                                blockRows,
                                filterType);

    filter.writeFilter(ncols, blockRows);

    //Assume Range sampling Frequency to be the same as 

      //ASSERT_LT(max_err, 1.0e-9);


}
/*
int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
*/

