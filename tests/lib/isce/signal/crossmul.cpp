
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
//int main(){
    //This test creates an interferogram between an SLC and itself and checks if the 
    //interferometric phase is zero.
    
    isce::io::Raster referenceSlc("/Users/fattahi/tools/ISCE3_forked/src/isce/tests/lib/isce/data/warped_envisat.slc.vrt");
    
    // make a raster of reference SLC
    //isce::io::Raster referenceSlc("../data/warped_envisat.slc.vrt");

    // a raster for the interferogram
    isce::io::Raster interferogram("igram.int", 500, 500, 1, GDT_CFloat32, "ISCE");

    std::string h5file("/Users/fattahi/tools/ISCE3_forked/src/isce/tests/lib/isce/data/envisat.h5");
    //std::string h5file("../data/envisat.h5");
    isce::io::IH5File file(h5file);
    isce::radar::Radar instrument;
    isce::radar::load(file, instrument);
    isce::core::Poly2d dop1 = instrument.contentDoppler();
    isce::core::Poly2d dop2 = instrument.contentDoppler();

    // Instantiate an ImageMode object
    isce::product::ImageMode mode;
    isce::product::load(file, mode, "aux");

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

    // get back the length and width of the interferogram 
    int width = interferogram.width();
    int length = interferogram.length();
    
    // an array for the ubterferogram
    std::valarray<std::complex<float>> data(width*length);

    // get the generated interferogram
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
    //
    isce::io::Raster referenceSlc("/Users/fattahi/tools/ISCE3_forked/src/isce/tests/lib/isce/data/warped_envisat.slc.vrt");

    // make a raster of reference SLC
    //isce::io::Raster referenceSlc("../data/warped_envisat.slc.vrt");
    int width = referenceSlc.width();
    int length = referenceSlc.length();
    std::valarray<std::complex<float>> geometryIgram(width*length);
    
    std::string h5file("/Users/fattahi/tools/ISCE3_forked/src/isce/tests/lib/isce/data/envisat.h5");
    //std::string h5file("../data/envisat.h5");
    isce::io::IH5File file(h5file);
    isce::radar::Radar instrument;
    isce::radar::load(file, instrument);
    isce::core::Poly2d dop1 = instrument.contentDoppler();
    isce::core::Poly2d dop2 = instrument.contentDoppler();

    // Instantiate an ImageMode object
    isce::product::ImageMode mode;
    isce::product::load(file, mode, "aux");

    double wvl = mode.wavelength();
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

    secSlc = refSlc*geometryPhase;

    std::valarray<std::complex<float>> Igram(width*length);
    for (size_t line = 0; line < length; ++line){
            for (size_t col = 0; col < width; ++col){
                Igram[line*width + col] =
                        refSlc[line*width + col]*
                        std::conj(secSlc[line*width + col]);
            }
    }

    isce::io::Raster simInterferogram("simulatedIgram.int", 500, 500, 1, GDT_CFloat32, "ISCE");
    simInterferogram.setBlock(Igram, 0 ,0 , width, length);

    secSlc = refSlc*geometryPhase;
    
    isce::io::Raster secondarySlc("secSlc.slc", 500, 500, 1, GDT_CFloat32, "ISCE");   
    secondarySlc.setBlock(secSlc, 0 ,0 , width, length);

    //instantiate the Crossmul class
    isce::signal::Crossmul crsmul;

    // set number of interferogram looks in range
    crsmul.rangeLooks(1);

    // set number of interferogram looks in azimuth
    crsmul.azimuthLooks(1);

    // set flag for performing common azimuthband filtering
    crsmul.doCommonAzimuthbandFiltering(false);

        
    isce::io::Raster interferogram("computedIgram.int", 500, 500, 1, GDT_CFloat32, "ISCE");
    // running crossmul
    crsmul.crossmul(referenceSlc, secondarySlc, interferogram); 

    

}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


