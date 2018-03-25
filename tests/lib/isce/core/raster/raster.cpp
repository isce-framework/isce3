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

#include "isce/core/Raster.h"

inline bool exists(const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

isce::core::Raster loadTestData();

struct RasterTest : public ::testing::Test {
  const uint nc = 100;  // number of columns
  const uint nl = 200;  // number of lines
  const uint nb = 5;    // block side
  const std::string latFilename = "lat.tif";
  const std::string lonFilename = "lon.rdr";
  const std::string incFilename = "inc.bin";
  isce::core::Raster img = loadTestData();
};


// Create GeoTiff Float dataset
TEST_F(RasterTest, createGeoTiffFloat) {
  std::remove(latFilename.c_str());
  isce::core::Raster lat = isce::core::Raster(latFilename, nc, nl, 1, GDT_Float32, "GTiff");
  ASSERT_EQ (exists(latFilename), true);
  ASSERT_EQ (lat.width(), nc);
  ASSERT_EQ (lat.length(), nl);
  ASSERT_EQ (lat.numBands(), 1);
  ASSERT_EQ (lat.dtype(), 6);  // GDT_Float32 = 6
}


// Create ISCE double dataset
TEST_F(RasterTest, createISCEDoubleSetValue) {
  std::remove(lonFilename.c_str());
  isce::core::Raster lon = isce::core::Raster(lonFilename, nc, nl, 1, GDT_Float64, "ISCE");
  ASSERT_EQ (exists(lonFilename), true);
  ASSERT_EQ (lon.dtype(), 7);  // GDT_Float64 = 7
  for ( uint y=0; y<lon.length(); ++y)
    for ( uint x=0; x<lon.width(); ++x)
      lon.setValue(y, x, y, 1);  // lon stores the line number in each pixel      
}


// GetValue from ISCE double dataset
TEST_F(RasterTest, openISCEDoubleGetValue) {
  double a;
  isce::core::Raster lon = isce::core::Raster(lonFilename);
  ASSERT_EQ (lon.access(), GA_ReadOnly);  // by default files are opened in readonly mode
  for (uint i=0; i<std::min(lon.width(), lon.length()); ++i) {
    lon.getValue(a, i, i, 1);
    ASSERT_EQ(a, i); // Diagonal elements must be equal
  }
}


// Update GeoTiff with set/get line and check valarray
TEST_F(RasterTest, updateGeoTiffFloat) {
  isce::core::Raster lat = isce::core::Raster(latFilename, GA_Update);
  std::vector<int> lineIn(nc);
  std::valarray<double> lineOut(nc);
  std::iota (std::begin(lineIn), std::end(lineIn), 0);
   for (uint y=0; y<lat.length(); ++y) {
    lat.setLine( lineIn, y );
    lat.getLine( lineOut, y );
    for (uint x=0; x<lat.width(); ++x)
      ASSERT_EQ ( lineOut[x], (double) x);
  }
}


// Create a 2-band file with ENVI format
TEST_F(RasterTest, createENVITwoBands) {
  std::remove(lonFilename.c_str());
  isce::core::Raster inc = isce::core::Raster(incFilename, nc, nl, 2, GDT_Int16);
  ASSERT_EQ (exists(incFilename), true);
  ASSERT_STREQ (inc.dataset()->GetDriverName(), isce::core::defaultGDALDriver.c_str());
}


// Populate first band ENVI file with setBlock
TEST_F(RasterTest, setblockENVIBandOne) {
  isce::core::Raster inc = isce::core::Raster(incFilename, GA_Update);
  //TODO: Check why it passes with GA_ReadOnly
  std::valarray<int> block( nb*nb );
  float a;
  for (uint y=0; y<ceil(inc.length()/nb); ++y) {
    for (uint x=0; x<ceil(inc.width()/nb); ++x) {
      block = x*y;
      inc.setBlock(block, x*nb, y*nb, nb, nb, 1);
      inc.getValue(a, x*nb+ceil(nb/2), y*nb+ceil(nb/2), 1);
      ASSERT_EQ (a, x*y);
    }
  }
}


// Populate second band ENVI file with setBlock
TEST_F(RasterTest, setblockENVIBandTwo) {
  isce::core::Raster inc = isce::core::Raster(incFilename, GA_Update);
  std::valarray<int> fullimg( nc*nl );
  std::valarray<int> block( 0, nb*nb );
  fullimg = 1;
  ASSERT_EQ (inc.numBands(), 2);
  inc.setBlock (fullimg, 0, 0, nc, nl, 2);
  inc.getBlock (block, 0, 0, nb, nb, 2);
  ASSERT_EQ ( block.sum(), nb*nb);
  //block = 0;
  //inc.getBlock (block, nc-2, nl-2, nb, nb, 2);
  //ASSERT_EQ ( block.sum(), nb*nb);
}




//////////////////
/* Check number of bands */
TEST_F(RasterTest, CheckNumBands) {
  const int refNumBands = 3;
  ASSERT_EQ(img.numBands(), refNumBands);
}







//////////////////////////////////
/* Check data types */
TEST_F(RasterTest, CheckDataType) {
  const std::vector<int> refDataType = {6, 7, 1};
  ASSERT_EQ(img.dataset()->GetRasterBand(1)->GetRasterDataType(), refDataType[0]); //Float32
  ASSERT_EQ(img.dataset()->GetRasterBand(2)->GetRasterDataType(), refDataType[1]); //Float64
  ASSERT_EQ(img.dataset()->GetRasterBand(3)->GetRasterDataType(), refDataType[2]); //Byte
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
  for (size_t i=0; i<img.width(); ++i) {
    img.getValue(a, i, i, 1);
    img.getValue(b, i, i, 2);
    img.getValue(c, i, i, 3);
    ASSERT_EQ(a, b);  // Diagonal elements in band 1 and 2 are equal
    ASSERT_EQ(c, 0.); // Diagonal elements in band 3 must be zero
  }
}


/* Check setValue */
TEST_F(RasterTest, CheckSetValue) {
  double a, b;
  int refValue = -1; // note the different type
  for (int i=0; i < (int) img.width(); ++i) {
    img.setValue(refValue, i, i, 1);
    img.getValue(a, i, i, 1);
    ASSERT_EQ(a, -1); // Diagonal elements are set to refValue and checked
    img.setValue (i, i, i, 1);
    img.getValue (a, i, i, 1);
    img.getValue (b, i, i, 2);
    ASSERT_EQ(a, b);  // Diagonal elements are re-set to row number and checked
  }
}
    
/* Check getLine */
TEST_F(RasterTest, CheckGetLine) {
  std::vector<float> line( img.width() );
  float sum     = 0.;
  float refSum  = 0.;
  float maskSum = 0.;
  
  for (size_t i=0; i<img.width(); ++i)
    refSum += i;
  
  for (size_t i=0; i<img.length(); ++i) {
    img.getLine( line, i, 1 );
    sum = std::accumulate( line.begin(), line.end(), 0. );
    ASSERT_EQ( sum, i*img.width());  // Band 1 has the row-number in each column

    img.getLine( line, i, 2 );
    sum = std::accumulate( line.begin(), line.end(), 0. );
    ASSERT_EQ( sum, refSum);   // Band 2 has the col-number in each column

    img.getLine( line, i, 3 );
    maskSum = std::accumulate( line.begin(), line.end(), maskSum );
  }
  
  ASSERT_EQ( maskSum, 0.5*img.width()*img.length());   // Band 3 is half zeros and half ones
}



/* Check getBlock */
TEST_F(RasterTest, CheckGetBlock) {

  int blockHeight = 21; // chosen so 300/21 = 14.285
  int blockWidth  = img.width();
  std::vector<float> block( blockWidth * blockHeight );
  std::vector<float> fullImage( img.width() * img.length() );
  float  sum     = 0.;
  float  refSum  = 0.;
  float  maskSum = 0.;
  size_t numBlocks = ceil( img.length() / blockHeight );
  
  for (size_t i=0; i<img.width(); ++i)
    refSum += i;
 
  for (size_t i=0; i<numBlocks; ++i) {
    img.getBlock (block, 0, 0, blockWidth, blockHeight, 2);
    sum = std::accumulate( block.begin(), block.end(), 0. );
    ASSERT_EQ( sum, refSum*blockHeight );  // Band 2 has the col-number in each column
  }

  img.getBlock (fullImage, 0, 0, img.width(), img.length(), 3);
  maskSum = std::accumulate( fullImage.begin(), fullImage.end(), 0. );
  ASSERT_EQ( maskSum, 0.5*img.width()*img.length());  // Band 3 is half zeros and half ones
}


/* CheckFileCreation */
TEST_F(RasterTest, CheckFileCreation) {
  double a;
  std::string fname = "created_testdata2.bin";
  isce::core::Raster img2 = isce::core::Raster(fname,
					       100, 200, 2, GDT_Float32, "ENVI");

  for (int i=0; i < (int) img2.width(); ++i) {
    img2.setValue( i, i, i, 1);
    img2.getValue(a, i, i, 1);
    ASSERT_EQ(i, a);
  }  
  std::remove(fname.c_str());
}

/* Check file creation using std::vector as input */
TEST_F(RasterTest, CheckFileCreationUsingStdVector) {
  double a;
  std::string fname = "created_testdata4.bin";
  std::vector<float> dataLine( 321 );
  isce::core::Raster img3 = isce::core::Raster(fname, dataLine, 400);

  for (int i=0; i < (int) img3.width(); ++i) {
    img3.setValue( i, i, i, 1);
    img3.getValue(a, i, i, 1);
    ASSERT_EQ(i, a);
  }
  std::remove(fname.c_str());
}


/* Check getLine */
TEST_F(RasterTest, CheckGetLineValarray) {
  std::valarray<float> line( img.width() );
  float sum     = 0.;
  float refSum  = 0.;
  float maskSum = 0.;
  
  for (size_t i=0; i<img.width(); ++i)
    refSum += i;
  
  for (size_t i=0; i<img.length(); ++i) {
    img.getLine( line, i, 1 );
    ASSERT_EQ( line.sum(), i*img.width());  // Band 1 has the row-number in each column
    img.getLine( line, i, 2 );
    ASSERT_EQ( line.sum(), refSum);   // Band 2 has the col-number in each column
    img.getLine( line, i, 3 );
    maskSum += line.sum();
  }
  ASSERT_EQ( maskSum, 0.5*img.width()*img.length());   // Band 3 is half zeros and half ones
}

TEST_F(RasterTest, createMultiBandVRT) {
  
  isce::core::Raster im1 = isce::core::Raster("test_data/cols_float64.bin", GA_ReadOnly);  
  std::cout << "reference :" << im1.dataset()->GetRefCount() << std::endl;
  isce::core::Raster im2 = isce::core::Raster("test_data/rows_float32.bin", GA_ReadOnly);  
  isce::core::Raster im_vrt = isce::core::Raster("test_data/test_multiband.vrt", {im1, im2});  
  std::cout << "reference :" << im1.dataset()->GetRefCount() << std::endl;
 
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

isce::core::Raster loadTestData() {

  isce::core::Raster img = isce::core::Raster("test_data/test_data.bin.vrt", GA_Update);

  return img;
}


// end of file
