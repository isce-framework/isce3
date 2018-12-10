#include <iostream>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include "isce/io/Raster.h"
#include <isce/signal/Looks.h>

TEST(Looks, realData)
{

    //shape of the array before multi-looking
    size_t width = 20;
    size_t length = 21; 

    // number of looks in range and azimuth (must not change for this unit test)
    size_t rngLooks = 3;
    size_t azLooks = 3;

    // shape of the multi-looked array
    size_t widthLooked = width/rngLooks;
    size_t lengthLooked = length/azLooks;

    // Buffers for original and multi-looked arrays
    std::valarray<float> x(width*length);
    std::valarray<float> xlks(widthLooked*lengthLooked);

    // fill the array
    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
              x[i*width + j] = i*j;
        }
    }

    //instantiate a looks object
    isce::signal::Looks<float> lksObj;
    lksObj.nrows(length);
    lksObj.ncols(width);
    lksObj.nrowsLooked(lengthLooked);
    lksObj.ncolsLooked(widthLooked);
    lksObj.rowsLooks(azLooks);
    lksObj.colsLooks(rngLooks);

    // multilook the x array and get the output in xlks array
    lksObj.multilook(x,xlks) ;

    //expected output for the multilooked array
    std::valarray<float> xlksExpected(widthLooked*lengthLooked);
    // Given the number of looks in range and azimuth (3x3) 
    // and given the array given above the multi-looked array
    // has the following values at the begining of each line
    // firstX = {1, 4, 7, 10, 13, 16, 19};
    // Also along the columns the following values are added to
    // first sample of the line. 
    // increment = {3 ,12 ,21 ,30 ,39 ,48 ,57};
    // So the expected multi-looked array is:

    for (size_t line = 0; line < lengthLooked; ++line){
        xlksExpected[line*widthLooked] = 1.0 + 3*line;
        float increment = 3*(1+line*3);
        for (size_t col = 1; col < widthLooked; ++col){
            
            xlksExpected[line*widthLooked + col] = xlksExpected[line*widthLooked]
                                            + col * increment; 
        }
    }


    float max_err = 0;
    float err = 0;
    for (size_t i = 0; i< widthLooked*lengthLooked; ++i){
        if (err > max_err)
            max_err = err;
    }

    ASSERT_LT(max_err, 1.0e-7);

}

int main(int argc, char * argv[]) {
      testing::InitGoogleTest(&argc, argv);
      return RUN_ALL_TESTS();
}



