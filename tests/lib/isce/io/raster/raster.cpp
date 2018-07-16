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
  const std::string latFilename = "lat.tif";
  const std::string lonFilename = "lon.vrt";
  const std::string incFilename = "inc.bin";
  const std::string mskFilename = "msk.bin";
  const std::string vrtFilename = "topo.vrt";
};


// Create GeoTiff one-band dataset (float)
TEST_F(RasterTest, createGeoTiffFloat) {
  std::remove(latFilename.c_str());
  isce::io::Raster lat = isce::io::Raster( latFilename, nc, nl, 1, GDT_Float32, "GTiff" );

  ASSERT_EQ( exists(latFilename), true);
  ASSERT_EQ( lat.width(),    nc );  // width must be number of columns
  ASSERT_EQ( lat.length(),   nl );  // length must be number of lines
  ASSERT_EQ( lat.numBands(), 1  );  // only one band has been created
  ASSERT_EQ( lat.dtype(),    6  );  // GDT_Float32 = 6
}



// Create VRT one-band dataset (double)
TEST_F(RasterTest, createVRTDouble_setGetValue) {
  std::remove(lonFilename.c_str());
  isce::io::Raster lon = isce::io::Raster( lonFilename, nc, nl, 1, GDT_Float64, "VRT" );
  float a;

  ASSERT_EQ( exists(lonFilename), true );  // check if exists
  ASSERT_EQ( lon.dtype(), 7 );             // GDT_Float64 = 7
  for ( uint y=0; y<lon.length(); ++y) {   // for each line
    for ( uint x=0; x<lon.width(); ++x) {  // for each column
      lon.setValue( y, x, y, 1);           // lon stores the line number in each pixel
      lon.getValue( a, x, y, 1);           // read double into a float
      ASSERT_EQ( a, y );                   // float a and uint y must be equal
    }
  }
}



// GetValue from VRT double dataset
TEST_F(RasterTest, openVRTRasterReadOnlyMode_getValue) {
  isce::io::Raster lon = isce::io::Raster( lonFilename );
  uint a;

  //ASSERT_EQ (lon.access(), GA_ReadOnly);  // files are opened in readonly mode by default
  for (uint i=0; i<std::min( nl, nc ); ++i) {
    lon.getValue( a, i, i, 1 );           // read double into a int
    ASSERT_EQ( a, i );                    // diagonal elements must be equal
  }
}


// GetValue from VRT double dataset opened as a GDAL dataset
TEST_F(RasterTest, openVRTGDALReadOnly_getValue) {
    // Open GDAL dataset
    GDALDataset * dset = static_cast<GDALDataset *>(
        GDALOpenShared(lonFilename.c_str(), GA_ReadOnly)
    );
    // Make an ISCE raster from the dataset
    isce::io::Raster lon = isce::io::Raster(dset);
    uint a;
    // Check values
    for (uint i = 0; i < std::min(nl, nc); ++i) {
        lon.getValue(a, i, i, 1);
        ASSERT_EQ(a, i);
    }
}


// Update GeoTiff line-wise and check valarray
TEST_F(RasterTest, updateGeoTiff_getLineValarray) {
  isce::io::Raster lat = isce::io::Raster( latFilename, GA_Update );
  std::vector<int>      lineIn(nc);
  std::valarray<double> lineOut(nc);
  std::iota (std::begin(lineIn), std::end(lineIn), 0);  // 0, 1, 2, ..., nc

  for ( uint y=0; y<lat.length(); ++y ) {  // for each line
    lat.setLine( lineIn,  y );             // set line from std::vector<int>
    lat.getLine( lineOut, y );             // get line into std::valarray<double>
    for ( uint x=0; x<lat.width(); ++x )   // for each column
      ASSERT_EQ( lineOut[x], (double) x);  // lat pixels store the column index
  }
}



// Create a 2-band file with ENVI format
TEST_F(RasterTest, createTwoBandsENVIRaster) {
  std::remove(incFilename.c_str());
  isce::io::Raster inc = isce::io::Raster(incFilename, nc, nl, 2, GDT_Int16, "ENVI");

  ASSERT_EQ( exists(incFilename), true );
}


// Populate first band of ENVI Raster with setBlock (blocks can't overflow image)
TEST_F(RasterTest, setBlockBandOneENVIRaster) {
  isce::io::Raster inc = isce::io::Raster( incFilename, GA_Update );
  std::valarray<int> block( nbx*nby );                           // 1d valarray for 2d blocks
  float a;
  uint nXBlocks = floor( (float) inc.width()  / (float) nbx );  // number of blocks along X
  uint nYBlocks = floor( (float) inc.length() / (float) nby );  // number of blocks along Y

  for ( uint y=0; y<nYBlocks; ++y ) {
    for ( uint x=0; x<nXBlocks; ++x ) {
      block = x*y;                                             // pick a value
      inc.setBlock( block, x*nbx, y*nby, nbx, nby, 1 );            // block must be within band
      inc.getValue( a, x*nbx+ceil(nbx/2.), y*nby+ceil(nby/2.), 1); // get block center value
      ASSERT_EQ( a, x*y );                                     // values must be equal
    }
  }
}


// Populate second band of ENVI Raster with setBlock (blocks can't overflow image)
TEST_F(RasterTest, setGetBlockBandTwoENVIRaster) {
  isce::io::Raster inc = isce::io::Raster(incFilename, GA_Update);
  std::valarray<int> fullimg( 1, nc*nl );       // ones
  std::valarray<int> block  ( 0, nbx*nby );       // zeros
  std::valarray<int> chunk  ( 9, (nbx+1) * (nby+1));   // nines

  ASSERT_EQ( inc.numBands(), 2 );               // inc must have two bands
  inc.setBlock( fullimg, 0, 0, nc, nl, 2 );     // write full image
  inc.getBlock( block, 0, 0, nbx, nby, 2);        // read upper-left sub-block
  inc.getBlock( chunk, 0, 0, nbx+1, nby+1, 1 );   // read chunk from band 1
  ASSERT_EQ( block.sum(), nbx*nby );              // block sum must be equal to nb^2
  ASSERT_EQ( chunk.sum(),  1);                  // chunk sum must be equal to 1
}



// Create VRT multiband from std::vector of Raster objects
TEST_F(RasterTest, createMultiBandVRT) {
  isce::io::Raster lat = isce::io::Raster( latFilename );
  isce::io::Raster lon = isce::io::Raster( lonFilename );
  isce::io::Raster inc = isce::io::Raster( incFilename );

  std::remove( vrtFilename.c_str() );
  ASSERT_EQ( lat.dataset()->GetRefCount(), 1 );  // one Raster points to lat
  isce::io::Raster vrt = isce::io::Raster( vrtFilename, {lat, lon, inc} );
  ASSERT_EQ( lat.dataset()->GetRefCount(), 2 );  // lat is now shared
}



// Check number of size and bands in input VRT
TEST_F(RasterTest, checkSizeNumBandsVRT) {
  isce::io::Raster vrt = isce::io::Raster( vrtFilename );
  const int refNumBands = 4;

  ASSERT_EQ( vrt.width(),  nc);              // number of columns
  ASSERT_EQ( vrt.length(), nl);              // number of lines
  ASSERT_EQ( vrt.numBands(), refNumBands );  // VRT must have 4 bands
}



// Check data types in multiband VRT dataset
TEST_F(RasterTest, checkDataTypeVRT) {
  isce::io::Raster vrt = isce::io::Raster( vrtFilename );
  const std::vector<int> refDataType = {6, 7, 3};

  ASSERT_EQ( vrt.dtype(1), refDataType[0] ); //Float32
  ASSERT_EQ( vrt.dtype(2), refDataType[1] ); //Float64
  ASSERT_EQ( vrt.dtype(4), refDataType[2] ); //Int
}



// Get lines from band 1 and 2 in VRT
TEST_F(RasterTest, checkGetLineVRT) {
  isce::io::Raster vrt = isce::io::Raster( vrtFilename );
  std::valarray<double> line( vrt.width() );
  double refSum  = 0.;

  for (size_t i=0; i<vrt.width(); ++i)
    refSum += i;

  for (size_t i=0; i<vrt.length(); ++i) {
    vrt.getLine( line, i, 1 );               // get line from band 1 in VRT
    ASSERT_EQ( line.sum(), refSum );         // lat has col-number in each pixel
    vrt.getLine( line, i, 2 );               // get line from band 2 in VRT
    ASSERT_EQ( line.sum(), i*vrt.width() );  // lon has row-number in each pixel
  }
}



// Create Raster using valarray or vector to infer width and type
TEST_F(RasterTest, createRasterFromStdVector) {
  std::valarray<uint8_t>  dataLineIn(  1,  nc );             // ones
  std::valarray<float>    dataLineOut( 0., nc );             // zeros
  isce::io::Raster msk = isce::io::Raster( mskFilename,  // filename
                                               dataLineIn,   // line valarray or vector
                                               nl );         // numnber of lines in Raster
    for (uint i=0; i < msk.width(); ++i) {
    msk.setLine( dataLineIn,  i );
    msk.getLine( dataLineOut, i );
    ASSERT_EQ( dataLineOut.sum(), (float) nc );              // sum of ones must be = nc
  }
}


// Add Raster to existing VRT and loop over all pixels in multiband VRT
TEST_F(RasterTest, addRasterToVRT) {
  isce::io::Raster vrt = isce::io::Raster( vrtFilename, GA_Update);
  isce::io::Raster msk = isce::io::Raster( mskFilename );
  uint refNumBands = 5;
  double val;

  vrt.addRasterToVRT( msk );                    // add all bands in msk to vrt
  ASSERT_EQ( vrt.numBands(), refNumBands);      // must be five due to previous tests
  ASSERT_EQ( vrt.dtype(5), GDT_Byte);           // must be uint8_t as per previous test

  for (uint b=1; b<=vrt.numBands(); ++b)        // for each 1-indexed band
    for (uint l=0; l<vrt.length(); ++l)         // for each 0-indexed line
      for (uint c=0; c<vrt.width(); ++c) {      // for each 0-indexed cols
        val = NAN;
        vrt.getValue ( val, c, l, b );          // get value for each pixel
        ASSERT_EQ( std::isfinite(val), true);   // value must be finite
      }
}



// Main
int main( int argc, char * argv[] ) {
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}


// end of file
