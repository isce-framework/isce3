//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Marco Lavalle
// Copyright 2018
//

#include <gtest/gtest.h>

#include "isce3/io/Raster.h"

// Support function to check if file exists
inline bool exists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}


// Global variables
class RasterTest : public ::testing::Test {
    public:
        const uint nc = 100;    // number of columns
        const uint nl = 200;    // number of lines
        const uint nbx = 5;      // block side length in x 
        const uint nby = 7;      //block size length in y 

    void SetUp(const std::string &filename) {
        std::remove( filename.c_str());
        isce3::io::Raster inc = isce3::io::Raster( filename, nc,
                                    nl, 1, GDT_Float32, "GTiff");
        std::valarray<int> block( nbx*nby );                           // 1d valarray for
        uint nXBlocks = floor( (float) inc.width()  / (float) nbx );  // number of blocks
        uint nYBlocks = floor( (float) inc.length() / (float) nby );  // number of blocks

        //Wrap Matrix around valarray
        isce3::core::Matrix<int> mat( &(block[0]), nby, nbx);

        for ( uint y=0; y<nYBlocks; ++y ) {
            for ( uint x=0; x<nXBlocks; ++x ) {
                block = x*y;                                 // pick a value
                inc.setBlock( mat, x*nbx, y*nby, 1);            // block must be within band
            }
        }
    }

};


// Populate first band of ENVI Raster with setBlock (blocks can't overflow image)
TEST_F(RasterTest, setBlockMatrix) {
    //Setup geotiff
    std::string filename = "matrix_gtiff.tif";
    SetUp(filename);

    isce3::io::Raster inc = isce3::io::Raster( filename, GA_ReadOnly);
    std::valarray<int> block( nbx*nby );                           // 1d valarray for 2d blocks
    float a;
    uint nXBlocks = floor( (float) inc.width()  / (float) nbx );  // number of blocks along X
    uint nYBlocks = floor( (float) inc.length() / (float) nby );  // number of blocks along Y

    for ( uint y=0; y<nYBlocks; ++y ) {
        for ( uint x=0; x<nXBlocks; ++x ) {
            block = x*y;                                 // pick a value
            inc.getValue( a, x*nbx+ceil(nbx/2.), y*nby+ceil(nby/2.), 1); // get block center value
            ASSERT_EQ( a, x*y );                                     // values must be equal
        }
    }
}


// Populate second band of ENVI Raster with setBlock (blocks can't overflow image)
TEST_F(RasterTest, setGetBlockMatrix) {
    //Setup geotiff
    std::string filename = "matrix_getset.tif";
    SetUp(filename);

    isce3::io::Raster inc = isce3::io::Raster(filename, GA_Update);
    std::valarray<int> fullimg( 1, nc*nl );       // ones
    std::valarray<int> block  ( 0, nbx*nby );       // zeros
    std::valarray<int> chunk  ( 9, (nbx+1) * (nby+1));   // nines

    //Wrap full image into matrix
    isce3::core::Matrix<int> fullmat(&(fullimg[0]), nl, nc);

    //Wrap block into Matrix
    isce3::core::Matrix<int> blockmat(&(block[0]), nby, nbx);

    //Wrap chunk into Matrix
    isce3::core::Matrix<int> chunkmat(&(chunk[0]), nby+1, nbx+1);

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
  isce3::core::Matrix<int> blockmat(nby, nbx);

  //Create raster object from matrix
  isce3::io::Raster raster(blockmat);

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
  isce3::core::Matrix<int> blockmat(nby, nbx);
  blockmat.fill(1);

  //Create raster object from matrix
  isce3::io::Raster raster(blockmat);

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


// 
TEST_F(RasterTest, setGetBlockEMatrix2D) {
  //Setup geotiff
  std::string filename = "matrix_getset.tif";
  SetUp(filename);

  isce3::io::Raster inc = isce3::io::Raster(filename, GA_Update);

  //Wrap full image into matrix
  isce3::core::EMatrix2D<int> fullmat(nl, nc);
  for (int i = 0; i < nl; i++ )
      for (int j = 0; j < nc; j++ )
        fullmat(i,j) = i*nc+j;

  //Wrap block into Matrix
  isce3::core::EMatrix2D<int> blockmat(nby, nbx);

  //Wrap chunk into Matrix
  isce3::core::EMatrix2D<int> chunkmat(nby+1, nbx+1);


  int block_mat_sum = 0;
  for (int i = 0; i < nby; i++ )
      for (int j = 0; j < nbx; j++ )
        block_mat_sum += i*nc+j;

  int chunk_mat_sum = 0;
  int l_start = nbx;
  int c_start = nbx;
  for (int  i = l_start; i < l_start+nby+1; i++ )
      for (int j = c_start; j < c_start + nbx+1; j++ )
        chunk_mat_sum += i*nc+j;

  ASSERT_EQ( inc.numBands(), 1 );               // inc must have one band
  inc.setBlock( fullmat, 0, 0, 1 );     // write full image
  inc.getBlock( blockmat, 0, 0, 1);        // read upper-left sub-block
  inc.getBlock( chunkmat, nbx, nbx, 1 );   // read chunk from band 1
  ASSERT_EQ(blockmat.sum(), block_mat_sum);              // block sum must be equal to nbx*nby
  ASSERT_EQ(chunkmat.sum(),  chunk_mat_sum);                  // chunk sum
}

TEST_F(RasterTest, setGetBlockEArray2D) {
  //Setup geotiff
  std::string filename = "matrix_getset.tif";
  SetUp(filename);

  isce3::io::Raster inc = isce3::io::Raster(filename, GA_Update);
    
  //Wrap full image into matrix
  isce3::core::EArray2D<int> fullmat(nl, nc);
  for (int i = 0; i < nl; i++ )
    for (int j = 0; j < nc; j++ )
        fullmat(i,j) = i*nc+j;

  //Wrap block into Matrix
  isce3::core::EArray2D<int> blockmat(nby, nbx);

  //Wrap chunk into Matrix
  isce3::core::EArray2D<int> chunkmat(nby+1, nbx+1);


  int block_mat_sum = 0;
  for (int i = 0; i < nby; i++ )
      for (int j = 0; j < nbx; j++ )
          block_mat_sum += i*nc+j;

  int chunk_mat_sum = 0;
  int l_start = nbx;
  int c_start = nbx;
  for (int  i = l_start; i < l_start+nby+1; i++ )
       for (int j = c_start; j < c_start + nbx+1; j++ )
          chunk_mat_sum += i*nc+j;

  ASSERT_EQ( inc.numBands(), 1 );               // inc must have one band
  inc.setBlock( fullmat, 0, 0, 1 );     // write full image
  inc.getBlock( blockmat, 0, 0, 1);        // read upper-left sub-block
  inc.getBlock( chunkmat, nbx, nbx, 1 );   // read chunk from band 1
  ASSERT_EQ( blockmat.sum(), block_mat_sum );              // block sum must be equal to nbx*nby
  ASSERT_EQ( chunkmat.sum(),  chunk_mat_sum);             // chunk sum

}

// Main
int main( int argc, char * argv[] ) {
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}


// end of file
