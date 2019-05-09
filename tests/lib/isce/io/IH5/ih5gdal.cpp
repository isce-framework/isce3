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
#include "gdal_alg.h"

// Output HDF5 file for file reading tests
std::string rFileName("../../data/envisat.h5");

struct IH5Test : public ::testing::Test {

   // Constructor
    protected:
        void SetUp()
        {
            GDALAllRegister();
            isce::io::GDALRegister_IH5();
        }

        void TearDown()
        {
            GDALDestroyDriverManager();
        }
};

TEST_F(IH5Test, dataSetOpen) {

    isce::io::IH5File file(rFileName);
    std::string datasetName("/science/LSAR/SLC/swaths/frequencyA/HH");

    isce::io::IDataSet dset = file.openDataSet(datasetName);
    auto dims = dset.getDimensions();
    std::string fname = dset.toGDAL();
        
    GDALDataset *ds = static_cast<GDALDataset*>(GDALOpen(fname.c_str(), GA_ReadOnly));

    //Check size and type
    ASSERT_EQ(ds->GetRasterXSize(), dims[1]);
    ASSERT_EQ(ds->GetRasterYSize(), dims[0]);
    ASSERT_EQ(ds->GetRasterCount(), 1);
    ASSERT_EQ(ds->GetRasterBand(1)->GetRasterDataType(), GDT_CFloat32);

    //Verify check sum
    int csum = GDALChecksumImage(ds->GetRasterBand(1), 0, 0, dims[1], dims[0]);
    ASSERT_EQ( csum, 4159);

    //Close datasets
    GDALClose(ds);
    dset.close();
}


TEST_F(IH5Test, dataSet3D) {
    isce::io::IH5File file(rFileName);
    std::string datasetName("/science/LSAR/SLC/metadata/geolocationGrid/incidenceAngle");

    isce::io::IDataSet dset = file.openDataSet(datasetName);
    auto dims = dset.getDimensions();
    std::string fname = dset.toGDAL();

    GDALDataset *ds = static_cast<GDALDataset*>(GDALOpen(fname.c_str(), GA_ReadOnly));

    //Check size and type
    ASSERT_EQ(ds->GetRasterXSize(), dims[2]);
    ASSERT_EQ(ds->GetRasterYSize(), dims[1]);
    ASSERT_EQ(ds->GetRasterCount(), dims[0]);
    ASSERT_EQ(ds->GetRasterBand(1)->GetRasterDataType(), GDT_Float32);

    //Close datasets
    GDALClose(ds);
    dset.close();
}


// Main
int main( int argc, char * argv[] ) {
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}


// end of file
