//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Marco Lavalle
// Copyright 2018
//

#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <numeric>
#include <gtest/gtest.h>

#include "isce/io/Raster.h"

// Support function to check if file exists
inline bool exists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}


// Global variables
struct RasterTest : public ::testing::Test {
  const uint nc = 11;    // number of columns
  const uint nl = 13;    // number of lines
  const uint nbx = 5;      // block side length in x 
  const uint nby = 7;      //block size length in y 
  const uint margin = 2;   //Extra padding to create matrix
  const std::string testFilename = "test.tif";
};


// Populate first band of ENVI Raster with setBlock (blocks can't overflow image)
TEST_F(RasterTest, setBlockView) {
  std::remove( testFilename.c_str());
  isce::io::Raster inc = isce::io::Raster( testFilename, nc, nl, 1, GDT_Float32, "GTiff" );

  isce::core::Matrix<float> fullmat(nl,nc);
  isce::core::Matrix<int> matrix(nby+margin, nbx+margin);
  float a;
   
  for(uint ii=0; ii < margin; ++ii)
      for(uint jj=0; jj < margin; ++jj)
      {
          //Ensure that the whole image is 1.
          fullmat.fill(1);
          inc.setBlock(fullmat, 0, 0);

          //Set full matrix to 2
          matrix.fill(2);
          auto view = matrix.submat(ii, jj, nby, nbx);

          //Set matrix using view at (1,1) in image
          inc.setBlock(view, 1, 1);

          //Get value at 1st element
          inc.getValue(a, 1, 1, 1);
          ASSERT_EQ(a,2);

          //Get value at last element
          inc.getValue(a, nbx, nby, 1);
          ASSERT_EQ(a,2);

          //Get value at mid element
          inc.getValue(a, (nbx+1)/2, (nby+1)/2, 1);
          ASSERT_EQ(a,2);

          //Check 0,0 is still 1
          inc.getValue(a, 0, 0, 1);
          ASSERT_EQ(a,1);
       }
}


// Populate first band of ENVI Raster with setBlock (blocks can't overflow image)
TEST_F(RasterTest, getSetBlockView) {
  std::remove( testFilename.c_str());
  isce::io::Raster inc = isce::io::Raster( testFilename, nc, nl, 1, GDT_Float32, "GTiff" );

  //Set the raster to 2
  isce::core::Matrix<float> fullmat(nl,nc);
  fullmat.fill(2);
  inc.setBlock(fullmat,0,0);

  //Create a larger matrix
  isce::core::Matrix<int> matrix(nl, nc);
  float a;
   
  for(uint ii=0; ii < margin; ++ii)
      for(uint jj=0; jj < margin; ++jj)
      {
          //Ensure that the whole matrix is 1.
          matrix.fill(1);

          //Create view. Ensure (0,0) is untouched for testing.
          auto view = matrix.submat(ii+1, jj+1, nby, nbx);

          //Read raster - all 2
          inc.getBlock(view, 0, 0);

          //Get value at 1st element
          ASSERT_EQ(matrix(ii+1, jj+1),2);

          //Get value at last element
          ASSERT_EQ(matrix(ii+nby,jj+nbx) ,2);

          //Get value at mid element
          ASSERT_EQ(matrix( ii + (nby+1)/2, jj + (nbx+1)/2), 2);

          //Check 0,0 is still 1
          ASSERT_EQ(matrix(0,0),1);
       }
}

// Main
int main( int argc, char * argv[] ) {
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}


// end of file
