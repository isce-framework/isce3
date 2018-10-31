
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

TEST(Crossmul, InterferogramZero)
{
    //This test creates an interferogram between an SLC and itself and checks if the phase is zero.
    //
    //isce::io::Raster refSlc("/Users/fattahi/tools/ISCE3_forked/src/isce/tests/lib/isce/data/warped_envisat.slc.vrt");
    // make a raster of reference SLC
    isce::io::Raster refSlc("../data/warped_envisat.slc.vrt");

    // a raster for the interferogram
    isce::io::Raster interferogram("igram.int", 500, 500, 1,
                                                        GDT_CFloat32, "ISCE");

    isce::signal::Crossmul crsmul;
    crsmul.crossmul(refSlc, refSlc, 1, 1, 10, interferogram);

     
    int width = interferogram.width();
    int length = interferogram.length();
    
    std::valarray<std::complex<float>> data(width*length);
    interferogram.getBlock(data, 0, 0, width, length);

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


