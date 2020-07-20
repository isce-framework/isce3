#include <gtest/gtest.h>
#include <string>

#include <isce3/except/Error.h>
#include <isce3/io/gdal/Dataset.h>
#include <isce3/io/gdal/Raster.h>

using isce::io::gdal::Dataset;
using isce::io::gdal::Raster;

struct DatasetTest : public testing::TestWithParam<std::string> {};

TEST_P(DatasetTest, Create)
{
    std::string driver = GetParam();
    std::string path = driver + "Dataset-create";
    int width = 8;
    int length = 4;
    int bands = 3;
    GDALDataType datatype = GDT_UInt16;

    Dataset dataset(path, width, length, bands, datatype, driver);

    EXPECT_EQ( dataset.access(), GA_Update );
    EXPECT_EQ( dataset.width(), width );
    EXPECT_EQ( dataset.length(), length );
    EXPECT_EQ( dataset.bands(), bands );
    EXPECT_EQ( dataset.driver(), driver );
    EXPECT_EQ( dataset.getRaster(1).datatype(), datatype );
}

TEST_P(DatasetTest, Copy)
{
    std::string driver = GetParam();
    std::string src_path = driver + "Dataset-src";
    std::string copy_path = driver + "Dataset-copy";
    int width = 5;
    int length = 6;
    int bands = 2;
    GDALDataType datatype = GDT_Float32;

    Dataset src(src_path, width, length, bands, datatype, driver);
    Dataset copy(copy_path, src);

    EXPECT_EQ( copy.width(), src.width() );
    EXPECT_EQ( copy.length(), src.length() );
    EXPECT_EQ( copy.bands(), src.bands() );
    EXPECT_EQ( copy.driver(), src.driver() );
    EXPECT_EQ( copy.getRaster(1).datatype(), src.getRaster(1).datatype() );
}

TEST_P(DatasetTest, GetRaster)
{
    std::string driver = GetParam();
    std::string path = driver + "Dataset-getraster";
    int width = 8;
    int length = 4;
    int bands = 3;
    GDALDataType datatype = GDT_Int32;

    Dataset dataset(path, width, length, bands, datatype, driver);

    for (int band = 1; band <= bands; ++band) {

        Raster raster = dataset.getRaster(band);

        EXPECT_EQ( raster.band(), band );
        EXPECT_EQ( raster.datatype(), datatype );
        EXPECT_EQ( raster.width(), dataset.width() );
        EXPECT_EQ( raster.length(), dataset.length() );
        EXPECT_EQ( raster.driver(), dataset.driver() );
        EXPECT_EQ( raster.getGeoTransform(), dataset.getGeoTransform() );
    }

    // attempting to fetch invalid raster band should throw
    {
        int band = bands + 1;
        EXPECT_THROW( { dataset.getRaster(band); }, isce::except::OutOfRange );
    }
}

// instantiate dataset tests for different drivers
INSTANTIATE_TEST_SUITE_P(ENVIDataset, DatasetTest, testing::Values("ENVI"));
INSTANTIATE_TEST_SUITE_P(GeoTiffDataset, DatasetTest, testing::Values("GTiff"));

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
