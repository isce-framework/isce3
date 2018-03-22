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

#include "isce/core/RasterLineIter.h"

isce::core::Raster loadTestData();

struct RasterTest : public ::testing::Test {
  
  isce::core::Raster img;
  isce::core::RasterLineIter lineIter = isce::core::RasterLineIter(img);
  
  protected:

  RasterTest() {
    img = loadTestData();
  }
};


TEST_F(RasterTest, createLineIter) {
  isce::core::RasterLineIter clonedLineIter(lineIter);
  // test copy constructor
  ASSERT_EQ(clonedLineIter, lineIter);
  lineIter++;
  lineIter--;
  // test increment/decrement
  ASSERT_EQ(clonedLineIter, lineIter);

}

TEST_F(RasterTest, lineIterRewind) {
  isce::core::RasterLineIter clonedLineIter(lineIter);  
  // test increment
  lineIter++;
  ASSERT_NE(clonedLineIter, lineIter);
  // test rewind
  lineIter.rewind();
  ASSERT_EQ(clonedLineIter, lineIter);
}

TEST_F(RasterTest, lineIterMath) {
  isce::core::RasterLineIter clonedLineIter(lineIter);  

  // test +=
  lineIter += 1;
  ASSERT_NE(clonedLineIter, lineIter);
  // test -=
  lineIter++;
  lineIter += 2;
  ASSERT_EQ(clonedLineIter, lineIter);  
}


    
/* Check getLine */
TEST_F(RasterTest, CheckGetNext) {

  std::vector<float> line( img.width() );
  float sum     = 0.;
  float refSum  = 0.;
  float maskSum = 0.;
  int i;
  
  for (size_t i=0; i<img.width(); i++)
    refSum += i;

  i = 0;
  while (lineIter != lineIter.atEnd()) {
    lineIter.getNext( line, 1 );    
    sum = std::accumulate( line.begin(), line.end(), 0. );
    // By construction band 1 has the row-number in each column
    ASSERT_EQ( sum, i*img.width());
    i++;
    
    lineIter.getNext( line, 2 );
    sum = std::accumulate( line.begin(), line.end(), 0. );
    // By construction band 2 has the col-number in each column
    ASSERT_EQ( sum, refSum);

    lineIter.getNext( line, 3 );
    maskSum = std::accumulate( line.begin(), line.end(), maskSum );
  }
  // By construction band 3 is half zeros and half ones
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
