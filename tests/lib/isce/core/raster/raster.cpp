//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Marco Lavalle
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <numeric>
#include <gtest/gtest.h>

#include "isce/core/Raster.h"

isce::core::Raster loadTestData();

struct RasterTest : public ::testing::Test {
  
  isce::core::Raster img;
  
  protected:

  RasterTest() {
    img = loadTestData();
  }
};

/* Check number of bands */
TEST_F(RasterTest, CheckNumBands) {

  const int refNumBands = 3;
  
  ASSERT_EQ(img.numBands(), refNumBands);
}


/* Check data types */
TEST_F(RasterTest, CheckDataType) {

  const std::vector<int> refDataType = {6, 7, 1};
  
  ASSERT_EQ(img.dataset->GetRasterBand(1)->GetRasterDataType(), refDataType[0]); //Float32
  ASSERT_EQ(img.dataset->GetRasterBand(2)->GetRasterDataType(), refDataType[1]); //Float64
  ASSERT_EQ(img.dataset->GetRasterBand(3)->GetRasterDataType(), refDataType[2]); //Byte
}


/* Check raster size */
TEST_F(RasterTest, CheckRasterSize) {

  const std::vector<int> refSize = {91, 300};
  
  ASSERT_EQ(img.width(), refSize[0]); // number of columns
  ASSERT_EQ(img.length(), refSize[1]); // number of lines
}


/* Check getValue */
TEST_F(RasterTest, CheckGetValue) {
  
  double a, b, c;

  for (size_t i=0; i<img.width(); i++) {
    img.getValue(a, i, i, 1);
    img.getValue(b, i, i, 2);
    img.getValue(c, i, i, 3);
    // Diagonal elements in band 1 and 2 must be equal
    ASSERT_EQ(a, b);
    // Diagonal elements in band 3 must be zero
    ASSERT_EQ(c, 0.);
  }
}


/* Check setValue */
TEST_F(RasterTest, CheckSetValue) {

  double a, b;
  int refValue = -1; // note the different type

  for (size_t i=0; i<img.width(); i++) {
    img.setValue(refValue, i, i, 1);
    img.getValue(a, i, i, 1);
    ASSERT_EQ(a, -1);

    img.setValue(i, i, i, 1);
    img.getValue(a, i, i, 1);
    img.getValue(b, i, i, 2);
    ASSERT_EQ(a, b);
  }
}
    
/* Check getLine */
TEST_F(RasterTest, CheckGetLine) {

  std::vector<float> line( img.width() );
  float sum     = 0.;
  float refSum  = 0.;
  float maskSum = 0.;
  
  for (size_t i=0; i<img.width(); i++)
    refSum += i;
  
  for (size_t i=0; i<img.length(); i++) {

    img.getLine( line, i, 1 );
    sum = std::accumulate( line.begin(), line.end(), 0. );
    // By construction band 1 has the row-number in each column
    ASSERT_EQ( sum, i*img.width());

    img.getLine( line, i, 2 );
    sum = std::accumulate( line.begin(), line.end(), 0. );
    // By construction band 2 has the col-number in each column
    ASSERT_EQ( sum, refSum);

    img.getLine( line, i, 3 );
    maskSum = std::accumulate( line.begin(), line.end(), maskSum );
  }
  // By construction band 3 is half zeros and half ones
  ASSERT_EQ( maskSum, 0.5*img.width()*img.length());
}



/* Check getBlock */
TEST_F(RasterTest, CheckGetBlock) {

  int blockHeight = 21; // chosen so 300/21 = 14.285
  int blockWidth  = img.width();
  std::vector<float> block( blockWidth * blockHeight );
  std::vector<float> fullImage( img.width() * img.length() );
  float sum     = 0.;
  float refSum  = 0.;
  float maskSum = 0.;
  
  size_t numBlocks = ceil( img.length() / blockHeight);
  
  for (size_t i=0; i<img.width(); i++)
    refSum += i;
 
  for (size_t i=0; i<numBlocks; i++) {
    img.getBlock (block, 0, 0, blockWidth, blockHeight, 2);
    sum = std::accumulate( block.begin(), block.end(), 0. );
    // By construction band 2 has the col-number in each column
    ASSERT_EQ( sum, refSum*blockHeight );
  }

  // By construction band 3 is half zeros and half ones
  img.getBlock (fullImage, 0, 0, img.width(), img.length(), 3);
  maskSum = std::accumulate( fullImage.begin(), fullImage.end(), 0. );
  ASSERT_EQ( maskSum, 0.5*img.width()*img.length());  
}



int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

isce::core::Raster loadTestData() {

  isce::core::Raster img = isce::core::Raster("test_data/test_data.bin.vrt", false);

  return img;
}


// end of file
