
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

    // HDF5 file with required metadata
    std::string h5file("../data/envisat.h5");
    
    //H5 object
    isce::io::IH5File file(h5file);

    // Open group containing instrument data
    isce::io::IGroup group = file.openGroup("/science/metadata/instrument_data");

    //Radar object and load the h5 file 
    isce::radar::Radar instrument;

    // Deserialize the radar instrument
    isce::radar::loadFromH5(group, instrument);

    // get the Doppler polynomial for refernce SLC
    isce::core::Poly2d dop1 = instrument.contentDoppler();

    // Since this test careates an interferogram between the refernce SLC and itself,
    // the second Doppler is the same as the first
    isce::core::Poly2d dop2 = instrument.contentDoppler();

    // Instantiate an ImageMode object
    isce::product::ImageMode mode;
    
    // Open group for image mode
    isce::io::IGroup modeGroup = file.openGroup("/science/complex_imagery");
    
    // Deserialize the primary_mode
    isce::product::loadFromH5(modeGroup, mode, "aux");

    // get the pulse repetition frequency (PRF)
    double prf = mode.prf();

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
    crsmul.crossmul(referenceSlc, referenceSlc, interferogram);

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

    // HDF5 file with required metadata
    std::string h5file("../data/envisat.h5");

    //H5 object
    isce::io::IH5File file(h5file);

    // Open group containing instrument data
    isce::io::IGroup group = file.openGroup("/science/metadata/instrument_data");

    //Radar object and load the h5 file
    isce::radar::Radar instrument;

    // Deserialize the radar instrument
    isce::radar::loadFromH5(group, instrument);

    // get the Doppler polynomial for refernce SLC
    isce::core::Poly2d dop1 = instrument.contentDoppler();

    // Since this test careates an interferogram between the refernce SLC and itself,
    // the second Doppler is the same as the first
    isce::core::Poly2d dop2 = instrument.contentDoppler();

    // Instantiate an ImageMode object
    isce::product::ImageMode mode;

    // Open group for image mode
    isce::io::IGroup modeGroup = file.openGroup("/science/complex_imagery");

    // Deserialize the primary_mode
    isce::product::loadFromH5(modeGroup, mode, "aux");
    

    //isce::product::load(file, mode, "aux");

    // get the pulse repetition frequency (PRF)
    double prf = mode.prf();

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
    crsmul.crossmul(referenceSlc, referenceSlc, interferogram);

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
         

TEST(Crossmul, CheckCrossmul)
{

    // a raster object for the reference SLC
    isce::io::Raster referenceSlc("../data/warped_envisat.slc.vrt");

    // get length and width
    int width = referenceSlc.width();
    int length = referenceSlc.length();

    std::string h5file("../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Open group containing instrument data
    isce::io::IGroup group = file.openGroup("/science/metadata/instrument_data");

    isce::radar::Radar instrument;

    // Deserialize the radar instrument
    isce::radar::loadFromH5(group, instrument);

    // extract the Doppler polynomial
    isce::core::Poly2d dop1 = instrument.contentDoppler();
    isce::core::Poly2d dop2 = instrument.contentDoppler();

    // Instantiate an ImageMode object
    isce::product::ImageMode mode;

    // Open group for image mode
    isce::io::IGroup modeGroup = file.openGroup("/science/complex_imagery");

    // Deserialize the primary_mode
    isce::product::loadFromH5(modeGroup, mode, "aux");


    // get the wavelength
    double wvl = mode.wavelength();

    // get range spacing
    double rngSpacing = mode.rangePixelSpacing();

    
    std::valarray<float> deltaR(width*length);
    std::valarray<std::complex<float>> geometryPhase(width*length);
    std::valarray<std::complex<float>> refSlc(width*length);
    std::valarray<std::complex<float>> secSlc(width*length);

    referenceSlc.getBlock(refSlc, 0 ,0, width, length);

    for (size_t line = 0; line < length; ++line){
        for (size_t col=0; col< width; ++col){
            deltaR[line*width + col] = rngSpacing * col /100;
            geometryPhase[line*width + col] = std::complex<float>(std::cos(deltaR[line*width + col]), std::sin(deltaR[line*width + col]));
        }
    }

    // phase of the second slc is the sum of phase of first slc and the geometryPhase
    secSlc = refSlc*geometryPhase;

    /*
    isce::io::Raster simInterferogram("simulatedIgram.int", width, length, 1, GDT_CFloat32, "ISCE");
    simInterferogram.setBlock(geometryPhase, 0 ,0 , width, length);
    */
    
    isce::io::Raster secondarySlc("/vsimem/secSlc.slc", width, length, 1, GDT_CFloat32, "ISCE");   
    secondarySlc.setBlock(secSlc, 0 ,0 , width, length);
    

    isce::io::Raster interferogram("computedIgram.int", width, length, 1, GDT_CFloat32, "ISCE");

    //instantiate the Crossmul class
    isce::signal::Crossmul crsmul;

    // set number of interferogram looks in range
    crsmul.rangeLooks(1);

    // set number of interferogram looks in azimuth
    crsmul.azimuthLooks(1);

    // set flag for performing common azimuthband filtering
    crsmul.doCommonAzimuthbandFiltering(false);

    // running crossmul
    crsmul.crossmul(referenceSlc, secondarySlc, interferogram); 

    // an array for the interferogram
    std::valarray<std::complex<float>> data(width*length);

    // get the generated interferogram
    interferogram.getBlock(data, 0, 0, width, length);
   

    
    double err = 0.0;
    double max_err = 0.0;
    int idx_max = 0;
    
    //don't evaluate the boundary of the interferogram for about 10 pixels
    int edge=20;
    
    for (size_t line = edge; line< length - edge; ++line){
	for (size_t col = edge; col< width - edge; ++col){
	    size_t i = line*width + col;
            err = std::arg(data[i]*(geometryPhase[i]));
            if (std::abs(err) > max_err){
                max_err = std::abs(err);
                idx_max = i;
            }
        }
    }

    // needs better understanding of the impact of oversampling
    // This threshold should be much smaller
    ASSERT_LT(max_err, 6.0e-1);
        

}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


