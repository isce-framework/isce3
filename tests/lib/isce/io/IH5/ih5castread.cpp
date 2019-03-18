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

#include "isce/io/IH5Dataset.h"
#include "isce/io/Raster.h"
#include "gdal_alg.h"


template<class A, class B>
struct TypeDefs
{
    typedef A firstType;
    typedef B secondType;
};


template<class T> 
struct IH5Test : public ::testing::Test {
    protected:
        IH5Test(){}

        void SetUp()
        {
            GDALAllRegister();
            GDALRegister_IH5();
        }

        void TearDown()
        {
            GDALDestroyDriverManager();
        }
};

//Types for which to test
typedef ::testing::Types<TypeDefs<unsigned char, short int>,
                         TypeDefs<unsigned char, unsigned short int>,
                         TypeDefs<unsigned char, int>,
                         TypeDefs<unsigned char, unsigned int>,
                         TypeDefs<unsigned char, float>,
                         TypeDefs<unsigned char, double>,
                         TypeDefs<unsigned char, std::complex<float>>,
                         TypeDefs<unsigned char, std::complex<double>>,
                         TypeDefs<short int, unsigned short int>,
                         TypeDefs<short int, int>,
                         TypeDefs<short int, unsigned int>,
                         TypeDefs<short int, float>,
                         TypeDefs<short int, double>,
                         TypeDefs<short int, std::complex<float>>,
                         TypeDefs<short int, std::complex<double>>,
                         TypeDefs<unsigned short int, int>,
                         TypeDefs<unsigned short int, unsigned int>,
                         TypeDefs<unsigned short int, float>,
                         TypeDefs<unsigned short int, double>,
                         TypeDefs<unsigned short int, std::complex<float>>,
                         TypeDefs<unsigned short int, std::complex<double>>,
                         TypeDefs<int, unsigned int>,
                         TypeDefs<int, float>,
                         TypeDefs<int, double>,
                         TypeDefs<int, std::complex<float>>,
                         TypeDefs<int, std::complex<double>>,
                         TypeDefs<unsigned int, float>,
                         TypeDefs<unsigned int, double>,
                         TypeDefs<unsigned int, std::complex<float>>,
                         TypeDefs<unsigned int, std::complex<double>>,
                         TypeDefs<float, double>,
                         TypeDefs<float, std::complex<float>>,
                         TypeDefs<float, std::complex<double>>,
                         TypeDefs<double, std::complex<double>>> MyTypes;

//Setup test suite
TYPED_TEST_CASE(IH5Test, MyTypes);

//This is a typed test
TYPED_TEST(IH5Test, nochunk) {

    //Typedefs for individual types
    typedef typename TypeParam::firstType FirstParam;
    typedef typename TypeParam::secondType SecondParam;

    //Create a matrix of typeparam
    int width = 20;
    int length = 30;
    isce::core::Matrix<FirstParam> _inmatrix(length, width);
    isce::core::Matrix<SecondParam> _outmatrix(length, width);
    for(size_t ii=0; ii< (width*length); ii++)
    {
        _inmatrix(ii) = (ii%255); 
        _outmatrix(ii) = (ii%255);
    }

    //Get checksum for the data
    int matsum;
    {
        isce::io::Raster matRaster(_outmatrix); 
        ASSERT_EQ( matRaster.dtype(1), isce::io::GDT.at(typeid(SecondParam)));
        matsum = GDALChecksumImage(matRaster.dataset()->GetRasterBand(1), 0, 0, width, length); 
    }

    //Create a HDF5 file
    std::string wfilename = "dummy.h5"; 
    struct stat buffer; 
    if ( stat(wfilename.c_str(), &buffer) == 0 ) 
        std::remove(wfilename.c_str()); 
    isce::io::IH5File fic(wfilename, 'x'); 
    
    isce::io::IGroup grp = fic.openGroup("/"); 
    std::array<int,2> shp={length, width};
    isce::io::IDataSet dset = grp.createDataSet(std::string("data"), _inmatrix.data(), shp); 
    {
        isce::io::Raster img(dset.toGDAL()); 
        
        //Check contents of the HDF5 file
        ASSERT_EQ( img.width(), width); 
        ASSERT_EQ( img.length(), length); 
        ASSERT_EQ( img.dtype(1), isce::io::GDT.at(typeid(FirstParam)));

        //Read data type with casting into another matrix
        //And compute check sum
        int hsum;
        {
            isce::core::Matrix<SecondParam> _readmatrix(length, width);
            img.getBlock(_readmatrix, 0, 0, 1);

            isce::io::Raster readRaster(_readmatrix);
            ASSERT_EQ( readRaster.dtype(1), isce::io::GDT.at(typeid(SecondParam)));
            hsum = GDALChecksumImage(readRaster.dataset()->GetRasterBand(1),0,0,width,length);
        }
        ASSERT_EQ( hsum, matsum);

        SecondParam val;
        img.getValue(val, 11, 13, 1);

        ASSERT_EQ( val, _outmatrix(13,11)); 
    }   
   
    //Cleanup
    dset.close();
    grp.close();
    fic.close();
    if ( stat(wfilename.c_str(), &buffer) == 0 )
        std::remove(wfilename.c_str());
}


//This is a typed test
TYPED_TEST(IH5Test, chunk) {

    //Typedefs for individual types
    typedef typename TypeParam::firstType FirstParam;
    typedef typename TypeParam::secondType SecondParam;

    //Create a matrix of typeparam
    int width = 250;
    int length = 200;
    isce::core::Matrix<FirstParam> _inmatrix(length, width);
    isce::core::Matrix<SecondParam> _outmatrix(length, width);
    for(size_t ii=0; ii< (width*length); ii++)
    {
        _inmatrix(ii) = (ii%255);
        _outmatrix(ii) = (ii%255);
    }

    //Get checksum for the data
    int matsum;
    {
        isce::io::Raster matRaster(_outmatrix);
        ASSERT_EQ( matRaster.dtype(1), isce::io::GDT.at(typeid(SecondParam)));
        matsum = GDALChecksumImage(matRaster.dataset()->GetRasterBand(1), 0, 0, width, length);
    }

    //Create a HDF5 file
    std::string wfilename = "dummy.h5";
    struct stat buffer;
    if ( stat(wfilename.c_str(), &buffer) == 0 )
        std::remove(wfilename.c_str());
    isce::io::IH5File fic(wfilename, 'x');

    isce::io::IGroup grp = fic.openGroup("/");
    std::array<int,2> shp={length, width};
    isce::io::IDataSet dset = grp.createDataSet<FirstParam>(std::string("data"), shp, 1);
    dset.write(_inmatrix.data(), width*length);
    {
        isce::io::Raster img(dset.toGDAL());

        //Check contents of the HDF5 file
        ASSERT_EQ( img.width(), width);
        ASSERT_EQ( img.length(), length);
        ASSERT_EQ( img.dtype(1), isce::io::GDT.at(typeid(FirstParam)));

        //Read data type with casting into another matrix
        //And compute check sum
        int hsum;
        {
            isce::core::Matrix<SecondParam> _readmatrix(length, width);
            img.getBlock(_readmatrix, 0, 0, 1);

            isce::io::Raster readRaster(_readmatrix);
            ASSERT_EQ( readRaster.dtype(1), isce::io::GDT.at(typeid(SecondParam)));
            hsum = GDALChecksumImage(readRaster.dataset()->GetRasterBand(1), 0, 0, width, length);
        }
        ASSERT_EQ( hsum, matsum);

        SecondParam val;
        //Quad 1
        img.getValue(val, 2, 3, 1);
        ASSERT_EQ( val, _outmatrix(3,2));

        //Quad 2
        img.getValue(val, 130, 5, 1);
        ASSERT_EQ(val, _outmatrix(5,130));

        //Quad 3
        img.getValue(val, 6, 129, 1);
        ASSERT_EQ(val, _outmatrix(129,6));

        //Quad 4
        img.getValue(val, 128, 135, 1);
        ASSERT_EQ( val, _outmatrix(135,128));
    }

    //Cleanup
    dset.close();
    grp.close();
    fic.close();
    if ( stat(wfilename.c_str(), &buffer) == 0 )
        std::remove(wfilename.c_str());
}


// Main
int main( int argc, char * argv[] ) {
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}


// end of file
