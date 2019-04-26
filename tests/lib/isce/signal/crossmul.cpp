
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
#include <isce/product/Product.h>
#include <isce/product/Serialization.h>


TEST(Crossmul, RunCrossmul)
{
    //This test creates an interferogram between an SLC and itself and checks if the 
    //interferometric phase is zero.
    
    //a raster object for the reference SLC
    isce::io::Raster referenceSlc("../data/warped_envisat.slc.vrt");

    // get the length and width of the SLC
    int width = referenceSlc.width();
    int length = referenceSlc.length();


    // a raster object for the interferogram
    isce::io::Raster interferogram("/vsimem/igram.int", width, length, 1, GDT_CFloat32, "ISCE");

    isce::io::Raster coherence("/vsimem/coherence.bin.", width, length, 1, GDT_Float32, "ISCE");
    // HDF5 file with required metadata
    std::string h5file("../data/envisat.h5");
    
    //H5 object
    isce::io::IH5File file(h5file);

    // Create a product and swath
    isce::product::Product product(file);
    const isce::product::Swath & swath = product.swath('A');

    // get the Doppler polynomial for refernce SLC
    isce::core::LUT1d<double> dop1 = product.metadata().procInfo().dopplerCentroid('A');

    // Since this test careates an interferogram between the refernce SLC and itself,
    // the second Doppler is the same as the first
    isce::core::LUT1d<double> dop2 = dop1;

    // get the pulse repetition frequency (PRF)
    double prf = swath.nominalAcquisitionPRF();

    //instantiate the Crossmul class  
    isce::signal::Crossmul crsmul;

    // set Doppler polynomials for refernce and secondary SLCs
    crsmul.doppler(dop1, dop2);

    // set prf
    crsmul.prf(prf);

    // set commonAzimuthBandwidth
    crsmul.commonAzimuthBandwidth(2000.0);
    
    // set beta parameter for cosine filter in commonAzimuthBandwidth filter
    crsmul.beta(0.25);

    // set number of interferogram looks in range
    crsmul.rangeLooks(1);

    // set number of interferogram looks in azimuth
    crsmul.azimuthLooks(1);

    // set flag for performing common azimuthband filtering
    crsmul.doCommonAzimuthbandFiltering(false);

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
          if (std::abs(err) > max_err){
              max_err = std::abs(err);
          }
      }

      ASSERT_LT(max_err, 1.0e-9);
}

TEST(Crossmul, RunCrossmulWithAzimuthCommonBandFilter)
{
    //This test creates an interferogram between an SLC and itself with azimuth
    //common band filtering and checks if the
    //interferometric phase is zero.

    //a raster object for the reference SLC
    isce::io::Raster referenceSlc("../data/warped_envisat.slc.vrt");

    // get the length and width of the SLC
    int width = referenceSlc.width();
    int length = referenceSlc.length();


    // a raster object for the interferogram
    isce::io::Raster interferogram("/vsimem/igram.int", width, length, 1, GDT_CFloat32, "ISCE");

    isce::io::Raster coherence("/vsimem/coherence.bin.", width, length, 1, GDT_Float32, "ISCE");

    // HDF5 file with required metadata
    std::string h5file("../data/envisat.h5");

    //H5 object
    isce::io::IH5File file(h5file);

    // Create a product and swath
    isce::product::Product product(file);
    const isce::product::Swath & swath = product.swath('A');

    // get the Doppler polynomial for refernce SLC
    isce::core::LUT1d<double> dop1 = product.metadata().procInfo().dopplerCentroid('A');

    // Since this test careates an interferogram between the refernce SLC and itself,
    // the second Doppler is the same as the first
    isce::core::LUT1d<double> dop2 = dop1;

    // get the pulse repetition frequency (PRF)
    double prf = swath.nominalAcquisitionPRF();

    //instantiate the Crossmul class
    isce::signal::Crossmul crsmul;

    // set Doppler polynomials for refernce and secondary SLCs
    crsmul.doppler(dop1, dop2);

    // set prf
    crsmul.prf(prf);

    // set commonAzimuthBandwidth
    crsmul.commonAzimuthBandwidth(2000.0);

    // set beta parameter for cosine filter in commonAzimuthBandwidth filter
    crsmul.beta(0.25);

    // set number of interferogram looks in range
    crsmul.rangeLooks(1);

    // set number of interferogram looks in azimuth
    crsmul.azimuthLooks(1);

    // set flag for performing common azimuthband filtering
    crsmul.doCommonAzimuthbandFiltering(true);

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
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
    }

    ASSERT_LT(max_err, 1.0e-9);
}
         


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


