#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include <isce3/core/EMatrix.h>
#include <isce3/io/Raster.h>
#include <isce3/signal/Looks.h>
#include <isce3/signal/multilook.h>

TEST(Looks, Multilook)
{
    // Note: this test is designed based on the following 
    // parameters (shape and number of looks). Changing these 
    // parameters requires subsequent changes to the code 
    // where the results are evaluated. 

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
    std::valarray<float> data(width*length);
    std::valarray<float> dataLooked(widthLooked*lengthLooked);
    std::valarray<float> dataLookednoData(widthLooked*lengthLooked);

    // Buffers for original and multi-looked complex data
    std::valarray<std::complex<float>> cpxData(width*length);
    std::valarray<std::complex<float>> cpxDataLooked(width*length);
    std::valarray<std::complex<float>> cpxDataLookednoData(width*length);

    // Same, but using Eigen datatypes
    isce3::core::EArray2D<float> a_data(length, width);
    isce3::core::EArray2D<std::complex<float>> a_cpxData(length, width);

    // fill the arrays
    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
              data[i*width + j] = i*j;
              a_data(i, j) = i * j;
              const std::complex<float> cpxval(std::cos(i * j),
                                               std::sin(i * j));
              cpxData[i * width + j] = cpxval;
              a_cpxData(i, j) = cpxval;
        }
    }

    //instantiate a looks object
    isce3::signal::Looks<float> lksObj;
    lksObj.nrows(length);
    lksObj.ncols(width);
    lksObj.nrowsLooked(lengthLooked);
    lksObj.ncolsLooked(widthLooked);
    lksObj.rowsLooks(azLooks);
    lksObj.colsLooks(rngLooks);

    // multilook the real data
    lksObj.multilook(data, dataLooked) ;
    const auto a_dataLooked =
            isce3::signal::multilookSummed(a_data, azLooks, rngLooks);

    // multilook the real data while excluding pixels with zero value.
    // This first creates a boolean mask and then creates a weight, 
    // which is 0 and 1.
    float noData = 0.0;
    lksObj.multilook(data, dataLookednoData, noData);
    const auto a_dataLookednoData =
            isce3::signal::multilookNoData(a_data, azLooks, rngLooks, noData);

    // multilook the complex data
    lksObj.multilook(cpxData, cpxDataLooked) ;
    const auto a_cpxDataLooked =
            isce3::signal::multilookSummed(a_cpxData, azLooks, rngLooks);

    // excluding pixels with 1 + 0.0J values (i.e., 1*exp(0.0j))
    std::complex<float>  cpxNoData = std::complex<float> (std::cos(0), std::sin(0));
    lksObj.multilook(cpxData, cpxDataLookednoData, cpxNoData) ;
    const auto a_cpxDataLookednoData = isce3::signal::multilookNoData(
            a_cpxData, azLooks, rngLooks, cpxNoData);

    // multilook the power of complex data (sum(abs(cpxData)^p))
    int p = 2;
    std::valarray<float> ampLooked(widthLooked*lengthLooked);
    lksObj.multilook(cpxData, ampLooked, p) ;
    const auto a_ampLooked =
            isce3::signal::multilookPow(a_cpxData, azLooks, rngLooks, p);

    //expected output for the multilooked array
    std::valarray<float> dataLookedExp(widthLooked*lengthLooked);
    isce3::core::EArray2D<float> a_dataLookedExp(lengthLooked, widthLooked);
    // Given the number of looks in range and azimuth (3x3) 
    // and given data above, the multi-looked array
    // has the following values at the beginning of each line
    // firstX = {1, 4, 7, 10, 13, 16, 19};
    // Also along the columns the following values (multiplied by column number) 
    // are added to first sample of the line. 
    // increment = {3 ,12 ,21 ,30 ,39 ,48 ,57};
    // So the expected multi-looked array is:

    for (size_t line = 0; line < lengthLooked; ++line){
        dataLookedExp[line*widthLooked] = 1.0 + 3*line;
        a_dataLookedExp(line, 0) = 1. + 3 * line;
        float increment = 3*(1+line*3);
        for (size_t col = 1; col < widthLooked; ++col){

            dataLookedExp[line*widthLooked + col] = dataLookedExp[line*widthLooked]
                                            + col * increment;
            a_dataLookedExp(line, col) =
                    a_dataLookedExp(line, 0) + col * increment;
        }
    }


    float max_err = 0;
    float err = 0;
    for (size_t i = 0; i< widthLooked*lengthLooked; ++i){
        err = std::abs(dataLookedExp[i] - dataLooked[i]); 
        if (err > max_err)
            max_err = err;
    }

    ASSERT_LT(max_err, 1.0e-6);

    max_err = (a_dataLookedExp - a_dataLooked).maxCoeff();
    EXPECT_LT(max_err, 1.0e-6);

    // check the phase of the first element of the multi-looked array
    ASSERT_NEAR(std::arg(cpxDataLooked[0]), 0.438899, 1.0e-6);
    EXPECT_NEAR(std::arg(a_cpxDataLooked(0, 0)), 0.438899, 1.0e-6);
    // check the phase of the last element of the multi-looked array
    ASSERT_NEAR(std::arg(cpxDataLooked[widthLooked*lengthLooked-1]), -0.880995, 1.0e-6);
    EXPECT_NEAR(std::arg(a_cpxDataLooked(lengthLooked - 1, widthLooked - 1)),
                -0.880995, 1.0e-6);

    // check the first element of the multi-looked amplitude
    ASSERT_NEAR(ampLooked[0], 9 / (rngLooks * azLooks), 1.0e-6);
    EXPECT_NEAR(a_ampLooked(0, 0), 9 / (rngLooks * azLooks), 1.0e-6);
    // check the last element of the multi-looked amplitude
    ASSERT_NEAR(ampLooked[widthLooked * lengthLooked - 1],
                9 / (rngLooks * azLooks), 1.0e-6);
    EXPECT_NEAR(a_ampLooked(lengthLooked - 1, widthLooked - 1),
                9 / (rngLooks * azLooks), 1.0e-6);

    // check the multi-looked real data when accounted for no data values
    // first element
    ASSERT_NEAR(dataLookednoData[0], 2.25, 1.0e-6);
    EXPECT_NEAR(a_dataLookednoData(0, 0), 2.25, 1.0e-6);
    // element [length-1, 0]
    ASSERT_NEAR(dataLookednoData[widthLooked*(lengthLooked-1)], 28.5, 1.0e-6);
    EXPECT_NEAR(a_dataLookednoData(lengthLooked - 1, 0), 28.5, 1.0e-6);

    // check the phase of multi-looked complex data when accounted for no data values
    // first element
    ASSERT_NEAR(std::arg(cpxDataLookednoData[0]), 2.031920, 1.0e-6);
    EXPECT_NEAR(std::arg(a_cpxDataLookednoData(0, 0)), 2.031920, 1.0e-6);
    // element [length-1, 0]
    ASSERT_NEAR(std::arg(cpxDataLookednoData[widthLooked*(lengthLooked-1)]), 0.161633, 1.0e-6);
    EXPECT_NEAR(std::arg(a_cpxDataLookednoData(lengthLooked - 1, 0)), 0.161633,
                1.0e-6);


}

int main(int argc, char * argv[]) {
      testing::InitGoogleTest(&argc, argv);
      return RUN_ALL_TESTS();
}
