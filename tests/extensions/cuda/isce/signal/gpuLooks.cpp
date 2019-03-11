#include <iostream>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include "isce/cuda/signal/gpuLooks.h"

struct gpuLooksTest : public ::testing::Test {
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

    // non complex original data
    std::valarray<float> data;

    // complex original data
    std::valarray<std::complex<float>> cpxData;

    isce::cuda::signal::gpuLooks<float> lksObj;

    protected:
        // constructor
        gpuLooksTest() {
            // instantiate a looks object
            isce::cuda::signal::gpuLooks<float> lksObj;
            lksObj.nrows(length);
            lksObj.ncols(width);
            lksObj.nrowsLooked(lengthLooked);
            lksObj.ncolsLooked(widthLooked);
            lksObj.rowsLooks(azLooks);
            lksObj.colsLooks(rngLooks);

            data.resize(width*length);
            cpxData.resize(width*length);

            // fill the arrays
            for (size_t i = 0; i< length; ++i){
                for (size_t j = 0; j< width; ++j){
                      data[i*width + j] = i*j;
                      cpxData[i*width + j] = std::complex<float> (std::cos(i*j), std::sin(i*j));
                }
            }
        }
};


TEST_F(gpuLooksTest, MultilookReal)
{
    // Buffer for multi-looked array
    std::valarray<float> dataLooked(widthLooked*lengthLooked);

    // multilook the real data
    lksObj.multilook(data, dataLooked) ;
    // Buffer for multi-looked array

    //expected output for the multilooked array
    std::valarray<float> dataLookedExp(widthLooked*lengthLooked);
    for (size_t line = 0; line < lengthLooked; ++line){
        dataLookedExp[line*widthLooked] = 1.0 + 3*line;
        float increment = 3*(1+line*3);
        for (size_t col = 1; col < widthLooked; ++col){
            
            dataLookedExp[line*widthLooked + col] = dataLookedExp[line*widthLooked]
                                            + col * increment; 
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
}


TEST_F(gpuLooksTest, MultilookNoDataReal)
{
    // Buffer for multi-looked array
    std::valarray<float> dataLookednoData(widthLooked*lengthLooked);

    // multilook the real data while excluding pixels with zero value.
    // This first creates a boolean mask and then creates a weight, 
    // which is 0 and 1.
    float noData = 0.0;
    lksObj.multilook(data, dataLookednoData, noData);
 
    // check the multi-looked real data when accounted for no data values
    // first element
    ASSERT_NEAR(dataLookednoData[0], 2.25, 1.0e-6);
    // element [length-1, 0]
    ASSERT_NEAR(dataLookednoData[widthLooked*(lengthLooked-1)], 28.5, 1.0e-6);
}


TEST_F(gpuLooksTest, MultilookComplex)
{
    // Buffer for multi-looked complex data
    std::valarray<std::complex<float>> cpxDataLooked(width*length);

    // multilook the complex data
    lksObj.multilook(cpxData, cpxDataLooked);

    // check the phase of the first element of the multi-looked array
    ASSERT_NEAR(std::arg(cpxDataLooked[0]), 0.438899, 1.0e-6);
    // check the phase of the last element of the multi-looked array
    ASSERT_NEAR(std::arg(cpxDataLooked[widthLooked*lengthLooked-1]), -0.880995, 1.0e-6);
}


TEST_F(gpuLooksTest, MultilookNoDataComplex)
{
    // Buffer for multi-looked complex data
    std::valarray<std::complex<float>> cpxDataLookednoData(width*length);

    // excluding pixels with 1 + 0.0J values (i.e., 1*exp(0.0j))
    std::complex<float>  cpxNoData = std::complex<float> (std::cos(0), std::sin(0));
    lksObj.multilook(cpxData, cpxDataLookednoData, cpxNoData);

    // check the phase of multi-looked complex data when accounted for no data values
    // first element
    ASSERT_NEAR(std::arg(cpxDataLookednoData[0]), 2.031920, 1.0e-6);
    // element [length-1, 0]
    ASSERT_NEAR(std::arg(cpxDataLookednoData[widthLooked*(lengthLooked-1)]), 0.161633, 1.0e-6);
}


TEST_F(gpuLooksTest, MultilookPower)
{
    // multilook the power of complex data (sum(abs(cpxData)^p))
    int p = 2;
    std::valarray<float> ampLooked(widthLooked*lengthLooked);
    lksObj.multilook(cpxData, ampLooked, p) ;

    // check the first element of the multi-looked amplitude
    ASSERT_NEAR(ampLooked[0], 9, 1.0e-6);
    // check the last element of the multi-looked amplitude
    ASSERT_NEAR(ampLooked[widthLooked*lengthLooked-1], 9, 1.0e-6);
}


int main(int argc, char **argv) {
      ::testing::InitGoogleTest(&argc, argv);
      return RUN_ALL_TESTS();
}



