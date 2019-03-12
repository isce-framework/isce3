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
  const uint nc = 100;    // number of columns
  const uint nl = 200;    // number of lines
  const uint nbx = 5;      // block side length in x 
  const uint nby = 7;      //block size length in y 
  const std::string testFilename = "test.tif";
};


// Populate first band of ENVI Raster with setBlock (blocks can't overflow image)
TEST_F(RasterTest, setBlockMatrix) {
  std::remove( testFilename.c_str());
  isce::io::Raster inc = isce::io::Raster( testFilename, nc, nl, 1, GDT_Float32, "GTiff" );
  std::valarray<int> block( nbx*nby );                           // 1d valarray for 2d blocks
  float a;
  uint nXBlocks = floor( (float) inc.width()  / (float) nbx );  // number of blocks along X
  uint nYBlocks = floor( (float) inc.length() / (float) nby );  // number of blocks along Y

  //Wrap Matrix around valarray
  isce::core::Matrix<int> mat( &(block[0]), nby, nbx);

  for ( uint y=0; y<nYBlocks; ++y ) {
    for ( uint x=0; x<nXBlocks; ++x ) {
      block = x*y;                                 // pick a value
      inc.setBlock( mat, x*nbx, y*nby, 1);            // block must be within band
      inc.getValue( a, x*nbx+ceil(nbx/2.), y*nby+ceil(nby/2.), 1); // get block center value
      ASSERT_EQ( a, x*y );                                     // values must be equal
    }
  }
}


// Populate second band of ENVI Raster with setBlock (blocks can't overflow image)
TEST_F(RasterTest, setGetBlockMatrix) {
  isce::io::Raster inc = isce::io::Raster(testFilename, GA_Update);
  std::valarray<int> fullimg( 1, nc*nl );       // ones
  std::valarray<int> block  ( 0, nbx*nby );       // zeros
  std::valarray<int> chunk  ( 9, (nbx+1) * (nby+1));   // nines

  //Wrap full image into matrix
  isce::core::Matrix<int> fullmat(&(fullimg[0]), nl, nc);

  //Wrap block into Matrix
  isce::core::Matrix<int> blockmat(&(block[0]), nby, nbx);

  //Wrap chunk into Matrix
  isce::core::Matrix<int> chunkmat(&(chunk[0]), nby+1, nbx+1);

  ASSERT_EQ( inc.numBands(), 1 );               // inc must have one band
  inc.setBlock( fullmat, 0, 0, 1 );     // write full image
  inc.getBlock( blockmat, 0, 0, 1);        // read upper-left sub-block
  inc.getBlock( chunkmat, nbx, nbx, 1 );   // read chunk from band 1
  ASSERT_EQ( block.sum(), nbx*nby );              // block sum must be equal to nbx*nby
  ASSERT_EQ( chunk.sum(),  (nbx+1)*(nby+1));                  // chunk sum 
}


//Test for Matrix behaving like a Raster
TEST_F(RasterTest, getMatrixRaster) {
  //Wrap block into Matrix
  isce::core::Matrix<int> blockmat(nby, nbx);

  //Create raster object from matrix
  isce::io::Raster raster(blockmat);

  //Fill the matrix
  for(uint ii=0; ii < nby; ii++)
      for(uint jj=0; jj < nbx; jj++)
          blockmat(ii,jj) = ii * nbx + jj;

  //Scalar for querying raster
  float a;
  raster.getValue(a,0,0,1);
  ASSERT_EQ( a, 0);
  raster.getValue(a, nbx-1, nby-1, 1);
  ASSERT_EQ( a, nbx*nby-1);
}


//Test for Matrix behaving like a Raster
TEST_F(RasterTest, setMatrixRaster) {
  //Wrap block into Matrix
  isce::core::Matrix<int> blockmat(nby, nbx);
  blockmat.fill(1);

  //Create raster object from matrix
  isce::io::Raster raster(blockmat);

  //Scalar for setting raster
  double a;

  //Set 0,0 and check
  a = 10.0;
  raster.setValue(a, 0, 0, 1);
  ASSERT_EQ(blockmat(0,0), 10);

  //Set last pixel
  a = -4.0;
  raster.setValue(a, nbx-1, nby-1, 1);
  ASSERT_EQ( blockmat(nby-1,nbx-1), -4);
}


// Main
int main( int argc, char * argv[] ) {
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}


// end of file
