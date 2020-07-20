#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <isce3/core/Projections.h>
#include <isce3/except/Error.h>
#include <isce3/io/gdal/Dataset.h>
#include <isce3/io/gdal/GeoTransform.h>
#include <isce3/io/gdal/Raster.h>
#include <isce3/io/IH5.h>

using isce3::core::ProjectionBase;
using isce3::io::gdal::Buffer;
using isce3::io::gdal::Dataset;
using isce3::io::gdal::GeoTransform;
using isce3::io::gdal::Raster;
using isce3::io::gdal::TypedBuffer;
using isce3::io::IH5File;
using isce3::io::IDataSet;

/** Raster w/ spatial reference & geo transform data */
struct DEMRasterTestData {
    std::string path;
    GDALDataType datatype;
    int length, width;
    double x0, y0, dx, dy;
    int epsg;
};

/**
 * 4x3 raster containing row major integer sequence from 0,...,11
 * w/ no spatial reference or geo transform data
 */
struct SequenceRasterTestData {
    std::string path;
};

struct RasterTestData {
    std::string driver;
    DEMRasterTestData dem;
    SequenceRasterTestData sequence;
};

/** Serialize RasterTestData to ostream */
std::ostream & operator<<(std::ostream & os, const RasterTestData & testdata)
{
    std::string out = testdata.driver + " test data";
    out += " (dem: " + testdata.dem.path + ", sequence: " + testdata.sequence.path + ")";
    return os << out;
}

struct RasterTest : public testing::TestWithParam<RasterTestData> {};

TEST_P(RasterTest, Open)
{
    RasterTestData testdata = GetParam();

    // default (read-only) access
    {
        Raster raster(testdata.dem.path);

        EXPECT_EQ( raster.band(), 1 );
        EXPECT_EQ( raster.datatype(), testdata.dem.datatype );
        EXPECT_EQ( raster.access(), GA_ReadOnly );
        EXPECT_EQ( raster.width(), testdata.dem.width );
        EXPECT_EQ( raster.length(), testdata.dem.length );
        EXPECT_EQ( raster.driver(), testdata.driver );
    }

    // read-write access
    {
        std::string path = testdata.driver + "Raster-open";
        int width = 4;
        int length = 8;
        GDALDataType datatype = GDT_Float32;

        // create new raster in working directory
        {
            Raster raster(path, width, length, datatype, testdata.driver);
        }

        Raster raster(path, GA_Update);

        EXPECT_EQ( raster.band(), 1 );
        EXPECT_EQ( raster.datatype(), datatype );
        EXPECT_EQ( raster.access(), GA_Update );
        EXPECT_EQ( raster.width(), width );
        EXPECT_EQ( raster.length(), length );
        EXPECT_EQ( raster.driver(), testdata.driver );
    }
}

TEST_P(RasterTest, OpenBand)
{
    RasterTestData testdata = GetParam();
    std::string path = testdata.driver + "Raster-openband";
    int width = 4;
    int length = 8;
    int bands = 2;
    GDALDataType datatype = GDT_UInt16;

    // create multi-band dataset
    {
        Dataset dataset(path, width, length, bands, datatype, testdata.driver);
    }

    // open the second raster band
    {
        int band = 2;
        Raster raster(path, band);

        EXPECT_EQ( raster.band(), band );
        EXPECT_EQ( raster.datatype(), datatype );
        EXPECT_EQ( raster.access(), GA_ReadOnly );
        EXPECT_EQ( raster.width(), width );
        EXPECT_EQ( raster.length(), length );
        EXPECT_EQ( raster.driver(), testdata.driver );
    }

    // attempting to fetch invalid raster band should throw
    {
        int band = bands + 1;
        EXPECT_THROW( { Raster raster(path, band); }, isce3::except::OutOfRange );
    }
}

TEST_P(RasterTest, Create)
{
    RasterTestData testdata = GetParam();
    std::string path = testdata.driver + "Raster-create";
    int width = 4;
    int length = 8;
    GDALDataType datatype = GDT_Float32;

    Raster raster(path, width, length, datatype, testdata.driver);

    EXPECT_EQ( raster.band(), 1 );
    EXPECT_EQ( raster.datatype(), datatype );
    EXPECT_EQ( raster.access(), GA_Update );
    EXPECT_EQ( raster.width(), width );
    EXPECT_EQ( raster.length(), length );
    EXPECT_EQ( raster.driver(), testdata.driver );
}

TEST_P(RasterTest, GetGeoTransform)
{
    RasterTestData testdata = GetParam();

    {
        Raster raster(testdata.dem.path);
        GeoTransform geo_transform = raster.getGeoTransform();

        EXPECT_DOUBLE_EQ( geo_transform.x0, testdata.dem.x0 );
        EXPECT_DOUBLE_EQ( geo_transform.y0, testdata.dem.y0 );
        EXPECT_DOUBLE_EQ( geo_transform.dx, testdata.dem.dx );
        EXPECT_DOUBLE_EQ( geo_transform.dy, testdata.dem.dy );

        EXPECT_EQ( geo_transform.x0, raster.x0() );
        EXPECT_EQ( geo_transform.y0, raster.y0() );
        EXPECT_EQ( geo_transform.dx, raster.dx() );
        EXPECT_EQ( geo_transform.dy, raster.dy() );
    }

    // raster with no geo transform data returns identity
    {
        Raster raster(testdata.sequence.path);
        GeoTransform geo_transform = raster.getGeoTransform();

        EXPECT_TRUE( geo_transform.isIdentity() );
    }
}

TEST_P(RasterTest, SetGeoTransform)
{
    RasterTestData testdata = GetParam();
    std::string path = testdata.driver + "Raster-setgeotransform";
    int width = 4;
    int length = 8;
    GDALDataType datatype = GDT_UInt16;

    GeoTransform geo_transform(1., 2., 3., -4.);

    {
        Raster raster(path, width, length, datatype, testdata.driver);
        raster.setGeoTransform(geo_transform);

        EXPECT_EQ( raster.getGeoTransform(), geo_transform );
    }

    // attempting to set geo transform for read-only raster should throw
    {
        Raster raster(path, GA_ReadOnly);

        EXPECT_THROW( { raster.setGeoTransform(geo_transform); }, isce3::except::RuntimeError );
    }
}

TEST_P(RasterTest, GetProjection)
{
    RasterTestData testdata = GetParam();

    {
        Raster raster(testdata.dem.path);
        std::unique_ptr<ProjectionBase> proj(raster.getProjection());

        EXPECT_EQ( proj->code(), testdata.dem.epsg );
    }

    // raster with no spatial reference system data
    {
        Raster raster(testdata.sequence.path);

        EXPECT_THROW( { raster.getProjection(); }, isce3::except::GDALError );
    }
}

TEST_P(RasterTest, SetProjection)
{
    RasterTestData testdata = GetParam();
    std::string path = testdata.driver + "Raster-setprojection";
    int width = 4;
    int length = 8;
    GDALDataType datatype = GDT_UInt16;

    std::unique_ptr<ProjectionBase> proj(isce3::core::createProj(4326));

    {
        Raster raster(path, width, length, datatype, testdata.driver);
        raster.setProjection(proj.get());

        std::unique_ptr<ProjectionBase> proj_out( raster.getProjection() );
        EXPECT_EQ( proj->code(), proj_out->code() );
    }

    // attempting to set projection for read-only raster should throw
    {
        Raster raster(path, GA_ReadOnly);

        EXPECT_THROW( { raster.setProjection(proj.get()); }, isce3::except::RuntimeError );
    }
}

TEST_P(RasterTest, ReadPixel)
{
    RasterTestData testdata = GetParam();

    Raster raster(testdata.sequence.path);

    int col = 2;
    int row = 2;
    int expected = 8;

    int val;
    raster.readPixel(&val, col, row);

    EXPECT_EQ( val, expected );
}

TEST_P(RasterTest, WritePixel)
{
    RasterTestData testdata = GetParam();
    std::string path = testdata.driver + "Raster-writepixel";
    int width = 4;
    int length = 8;
    GDALDataType datatype = GDT_Int32;

    int col = 3;
    int row = 4;
    int expected = 123;

    {
        Raster raster(path, width, length, datatype, testdata.driver);
        raster.writePixel(&expected, col, row);

        int val;
        raster.readPixel(&val, col, row);

        EXPECT_EQ( val, expected );
    }

    // writing to read-only raster should throw
    {
        Raster raster(path, GA_ReadOnly);
        EXPECT_THROW( { raster.writePixel(&expected, col, row); }, isce3::except::RuntimeError );
    }
}

TEST_P(RasterTest, ReadLine)
{
    RasterTestData testdata = GetParam();

    Raster raster(testdata.sequence.path);

    int row = 1;
    std::vector<int> expected = { 3, 4, 5 };

    std::vector<int> vals(raster.width());
    raster.readLine(vals.data(), row);

    EXPECT_EQ( vals, expected );
}

TEST_P(RasterTest, WriteLine)
{
    RasterTestData testdata = GetParam();
    std::string path = testdata.driver + "Raster-writeline";
    int width = 4;
    int length = 8;
    GDALDataType datatype = GDT_Int32;

    int row = 4;
    std::vector<int> expected = { 10, 11, 12, 13 };

    {
        Raster raster(path, width, length, datatype, testdata.driver);
        raster.writeLine(expected.data(), row);

        std::vector<int> vals(raster.width());
        raster.readLine(vals.data(), row);

        EXPECT_EQ( vals, expected );
    }

    // writing to read-only raster should throw
    {
        Raster raster(path, GA_ReadOnly);
        EXPECT_THROW( { raster.writeLine(expected.data(), row); }, isce3::except::RuntimeError );
    }
}

TEST_P(RasterTest, ReadLines)
{
    RasterTestData testdata = GetParam();

    Raster raster(testdata.sequence.path);

    int first_row = 0;
    int num_rows = 2;
    std::vector<int> expected = { 0, 1, 2,
                                  3, 4, 5 };

    std::vector<int> vals(num_rows * raster.width());
    raster.readLines(vals.data(), first_row, num_rows);

    EXPECT_EQ( vals, expected );
}

TEST_P(RasterTest, WriteLines)
{
    RasterTestData testdata = GetParam();
    std::string path = testdata.driver + "Raster-writelines";
    int width = 4;
    int length = 8;
    GDALDataType datatype = GDT_Int32;

    int first_row = 2;
    int num_rows = 3;
    std::vector<int> expected = {  1,  2,  3,  4,
                                   8,  7,  6,  5,
                                  -1, -2, -3, -4 };

    {
        Raster raster(path, width, length, datatype, testdata.driver);
        raster.writeLines(expected.data(), first_row, num_rows);

        std::vector<int> vals(num_rows * raster.width());
        raster.readLines(vals.data(), first_row, num_rows);

        EXPECT_EQ( vals, expected );
    }

    // writing to read-only raster should throw
    {
        Raster raster(path, GA_ReadOnly);
        EXPECT_THROW( { raster.writeLines(expected.data(), first_row, num_rows); }, isce3::except::RuntimeError );
    }
}

TEST_P(RasterTest, ReadBlock)
{
    RasterTestData testdata = GetParam();

    Raster raster(testdata.sequence.path);

    int first_col = 1;
    int first_row = 0;
    int num_cols = 2;
    int num_rows = 4;
    std::vector<int> expected = {  1,  2,
                                   4,  5,
                                   7,  8,
                                  10, 11 };

    std::vector<int> vals(num_cols * num_rows);
    raster.readBlock(vals.data(), first_col, first_row, num_cols, num_rows);

    EXPECT_EQ( vals, expected );
}

TEST_P(RasterTest, WriteBlock)
{
    RasterTestData testdata = GetParam();
    std::string path = testdata.driver + "Raster-writeblock";
    int width = 4;
    int length = 8;
    GDALDataType datatype = GDT_Int32;

    int first_col = 0;
    int first_row = 1;
    int num_cols = 2;
    int num_rows = 2;
    std::vector<int> expected = { 1, 2,
                                  3, 4 };

    {
        Raster raster(path, width, length, datatype, testdata.driver);
        raster.writeBlock(expected.data(), first_col, first_row, num_cols, num_rows);

        std::vector<int> vals(num_cols * num_rows);
        raster.readBlock(vals.data(), first_col, first_row, num_cols, num_rows);

        EXPECT_EQ( vals, expected );
    }

    // writing to read-only raster should throw
    {
        Raster raster(path, GA_ReadOnly);
        EXPECT_THROW(
                { raster.writeBlock(expected.data(), first_col, first_row, num_cols, num_rows); },
                isce3::except::RuntimeError );
    }
}

TEST_P(RasterTest, ReadAll)
{
    RasterTestData testdata = GetParam();

    Raster raster(testdata.sequence.path);

    std::vector<int> expected = {  0,  1,  2,
                                   3,  4,  5,
                                   6,  7,  8,
                                   9, 10, 11 };

    std::vector<int> vals(raster.length() * raster.width());
    raster.readAll(vals.data());

    EXPECT_EQ( vals, expected );
}

TEST_P(RasterTest, WriteAll)
{
    RasterTestData testdata = GetParam();
    std::string path = testdata.driver + "Raster-writeall";
    int width = 3;
    int length = 2;
    GDALDataType datatype = GDT_Int32;

    std::vector<int> expected = { 1, 2, 3,
                                  4, 5, 6 };

    {
        Raster raster(path, width, length, datatype, testdata.driver);
        raster.writeAll(expected.data());

        std::vector<int> vals(raster.length() * raster.width());
        raster.readAll(vals.data());

        EXPECT_EQ( vals, expected );
    }

    // writing to read-only raster should throw
    {
        Raster raster(path, GA_ReadOnly);
        EXPECT_THROW( { raster.writeAll(expected.data()); }, isce3::except::RuntimeError );
    }
}

TEST_P(RasterTest, OutOfBoundsRead)
{
    RasterTestData testdata = GetParam();

    Raster raster(testdata.sequence.path);

    int col = raster.width() + 1;
    int row = 0;

    int val;
    EXPECT_THROW( { raster.readPixel(&val, col, row); }, isce3::except::OutOfRange );
}

TEST_P(RasterTest, OutOfBoundsWrite)
{
    RasterTestData testdata = GetParam();
    std::string path = testdata.driver + "Raster-outofboundswrite";
    int width = 4;
    int length = 8;
    GDALDataType datatype = GDT_Int32;
    Raster raster(path, width, length, datatype, testdata.driver);

    int col = width - 1;
    int row = -1;

    int val = 123;
    EXPECT_THROW( { raster.writePixel(&val, col, row); }, isce3::except::OutOfRange );
}

TEST_P(RasterTest, Memmap)
{
    RasterTestData testdata = GetParam();
    Raster raster(testdata.sequence.path);

    {
        Buffer mmap = raster.memmap();

        using T = std::int32_t;
        T * data = static_cast<T *>(mmap.data());
        for (int i = 0; i < raster.length() * raster.width(); ++i) {
            EXPECT_EQ( data[i], T(i) );
        }

        EXPECT_EQ( mmap.datatype(), raster.datatype() );
        EXPECT_EQ( mmap.length(), raster.length() );
        EXPECT_EQ( mmap.width(), raster.width() );
        EXPECT_EQ( mmap.colstride(), sizeof(T) );
        EXPECT_EQ( mmap.rowstride(), raster.width() * sizeof(T) );
    }

    {
        using T = std::int32_t;
        TypedBuffer<T> mmap = raster.memmap<T>();

        T * data = mmap.data();
        for (int i = 0; i < raster.length() * raster.width(); ++i) {
            EXPECT_EQ( data[i], T(i) );
        }

        EXPECT_EQ( mmap.datatype(), raster.datatype() );
        EXPECT_EQ( mmap.length(), raster.length() );
        EXPECT_EQ( mmap.width(), raster.width() );
        EXPECT_EQ( mmap.colstride(), sizeof(T) );
        EXPECT_EQ( mmap.rowstride(), raster.width() * sizeof(T) );
    }
}

RasterTestData envi_test_data {
        "ENVI", // driver
        { // dem
            TESTDATA_DIR "io/gdal/ENVIRaster-dem", // path
            GDT_Float32, // datatype
            72, // length
            36, // width
            -156.000138888886, // x0
            20.0001388888836, // y0
            0.000277777777777815, // dx
            -0.000277777777777815, // dy
            4326 // epsg
        },
        { // sequence
            TESTDATA_DIR "io/gdal/ENVIRaster-sequence" // path
        }};

RasterTestData geotiff_test_data {
        "GTiff", // driver
        { // dem
            TESTDATA_DIR "io/gdal/GTiffRaster-dem", // path
            GDT_Float32, // datatype
            72, // length
            36, // width
            -156.000138888886, // x0
            20.0001388888836, // y0
            0.000277777777777815, // dx
            -0.000277777777777815, // dy
            4326 // epsg
        },
        { // sequence
            TESTDATA_DIR "io/gdal/GTiffRaster-sequence" // path
        }};

// instantiate raster tests for different drivers
INSTANTIATE_TEST_SUITE_P(ENVIRaster, RasterTest, testing::Values(envi_test_data));
INSTANTIATE_TEST_SUITE_P(GeoTiffRaster, RasterTest, testing::Values(geotiff_test_data));

struct H5RasterTest : public testing::Test {
    std::string filepath = TESTDATA_DIR "io/gdal/IH5Raster.h5";
    std::string datasetpath = "sequence";
    GDALDataType datatype = GDT_Int32;
    int width = 3;
    int length = 4;
    std::string driver = "IH5";
};

TEST_F(H5RasterTest, Open)
{
    IH5File file(filepath);
    IDataSet dataset = file.openDataSet(datasetpath);

    Raster raster(dataset);

    EXPECT_EQ( raster.band(), 1 );
    EXPECT_EQ( raster.datatype(), datatype );
    EXPECT_EQ( raster.access(), GA_ReadOnly );
    EXPECT_EQ( raster.width(), width );
    EXPECT_EQ( raster.length(), length );
    EXPECT_EQ( raster.driver(), driver );
}

TEST_F(H5RasterTest, OpenBand)
{
    IH5File file(filepath);
    IDataSet dataset = file.openDataSet(datasetpath);

    {
        int band = 1;
        Raster raster(dataset, band);

        EXPECT_EQ( raster.band(), band );
        EXPECT_EQ( raster.datatype(), datatype );
        EXPECT_EQ( raster.access(), GA_ReadOnly );
        EXPECT_EQ( raster.width(), width );
        EXPECT_EQ( raster.length(), length );
        EXPECT_EQ( raster.driver(), driver );
    }

    // attempting to fetch invalid raster band should throw
    {
        int band = 100;
        EXPECT_THROW( { Raster raster(dataset, band); }, isce3::except::OutOfRange );
    }
}

TEST_F(H5RasterTest, ReadPixel)
{
    IH5File file(filepath);
    IDataSet dataset = file.openDataSet(datasetpath);

    Raster raster(dataset);

    int col = 2;
    int row = 2;
    int expected = 8;

    int val;
    raster.readPixel(&val, col, row);

    EXPECT_EQ( val, expected );
}

TEST_F(H5RasterTest, ReadLine)
{
    IH5File file(filepath);
    IDataSet dataset = file.openDataSet(datasetpath);

    Raster raster(dataset);

    int row = 1;
    std::vector<int> expected = { 3, 4, 5 };

    std::vector<int> vals(raster.width());
    raster.readLine(vals.data(), row);

    EXPECT_EQ( vals, expected );
}

TEST_F(H5RasterTest, ReadLines)
{
    IH5File file(filepath);
    IDataSet dataset = file.openDataSet(datasetpath);

    Raster raster(dataset);

    int first_row = 0;
    int num_rows = 2;
    std::vector<int> expected = { 0, 1, 2,
                                  3, 4, 5 };

    std::vector<int> vals(num_rows * raster.width());
    raster.readLines(vals.data(), first_row, num_rows);

    EXPECT_EQ( vals, expected );
}

TEST_F(H5RasterTest, ReadBlock)
{
    IH5File file(filepath);
    IDataSet dataset = file.openDataSet(datasetpath);

    Raster raster(dataset);

    int first_col = 1;
    int first_row = 0;
    int num_cols = 2;
    int num_rows = 4;
    std::vector<int> expected = {  1,  2,
                                   4,  5,
                                   7,  8,
                                  10, 11 };

    std::vector<int> vals(num_cols * num_rows);
    raster.readBlock(vals.data(), first_col, first_row, num_cols, num_rows);

    EXPECT_EQ( vals, expected );
}

TEST_F(H5RasterTest, ReadAll)
{
    IH5File file(filepath);
    IDataSet dataset = file.openDataSet(datasetpath);

    Raster raster(dataset);

    std::vector<int> expected = {  0,  1,  2,
                                   3,  4,  5,
                                   6,  7,  8,
                                   9, 10, 11 };

    std::vector<int> vals(raster.length() * raster.width());
    raster.readAll(vals.data());

    EXPECT_EQ( vals, expected );
}

struct MEMRasterTest : public testing::Test {
    std::string driver = "MEM";
    int width = 3;
    int length = 4;
    std::vector<int> v;

    void SetUp() override
    {
        v = std::vector<int>(length * width);
    }
};

TEST_F(MEMRasterTest, SimpleDataLayout)
{
    // raster is read-only if data pointer is const
    {
        const int * data = v.data();
        Raster raster(data, width, length);

        EXPECT_EQ( raster.band(), 1 );
        EXPECT_EQ( raster.access(), GA_ReadOnly );
        EXPECT_EQ( raster.width(), width );
        EXPECT_EQ( raster.length(), length );
        EXPECT_EQ( raster.driver(), driver );
    }

    // non-const pointer, raster can be read-only or read/write
    {
        int * data = v.data();
        Raster raster(data, width, length, GA_Update);

        EXPECT_EQ( raster.band(), 1 );
        EXPECT_EQ( raster.access(), GA_Update );
        EXPECT_EQ( raster.width(), width );
        EXPECT_EQ( raster.length(), length );
        EXPECT_EQ( raster.driver(), driver );
    }
}

TEST_F(MEMRasterTest, AdvancedDataLayout)
{
    std::size_t colstride = sizeof(int);
    std::size_t rowstride = width * sizeof(int);

    // raster is read-only if data pointer is const
    {
        const int * data = v.data();
        Raster raster(data, width, length, colstride, rowstride);

        EXPECT_EQ( raster.band(), 1 );
        EXPECT_EQ( raster.access(), GA_ReadOnly );
        EXPECT_EQ( raster.width(), width );
        EXPECT_EQ( raster.length(), length );
        EXPECT_EQ( raster.driver(), driver );
    }

    // non-const pointer, raster can be read-only or read/write
    {
        int * data = v.data();
        Raster raster(data, width, length, colstride, rowstride, GA_Update);

        EXPECT_EQ( raster.band(), 1 );
        EXPECT_EQ( raster.access(), GA_Update );
        EXPECT_EQ( raster.width(), width );
        EXPECT_EQ( raster.length(), length );
        EXPECT_EQ( raster.driver(), driver );
    }
}

TEST_F(MEMRasterTest, ReadBlockRowMajor)
{
    v = {  0,  1,  2,
           3,  4,  5,
           6,  7,  8,
           9, 10, 11 };

    Raster raster(v.data(), width, length);

    std::vector<int> vals(length * width);
    raster.readBlock(vals.data(), 0, 0, width, length);

    EXPECT_EQ( vals, v );
}

TEST_F(MEMRasterTest, WriteBlockRowMajor)
{
    Raster raster(v.data(), width, length);

    std::vector<int> vals = {  0,  1,  2,
                               3,  4,  5,
                               6,  7,  8,
                               9, 10, 11 };

    raster.writeBlock(vals.data(), 0, 0, width, length);

    EXPECT_EQ( v, vals );

    // writing to read-only raster should throw
    {
        Raster raster(v.data(), width, length, GA_ReadOnly);
        EXPECT_THROW( { raster.writeBlock(vals.data(), 0, 0, width, length); }, isce3::except::RuntimeError );
    }
}

TEST_F(MEMRasterTest, ReadBlockColMajor)
{
    v = {  0,  3,  6,  9,
           1,  4,  7, 10,
           2,  5,  8, 11 };

    std::size_t rowstride = sizeof(int);
    std::size_t colstride = length * rowstride;
    Raster raster(v.data(), width, length, colstride, rowstride);

    std::vector<int> vals(length * width);
    raster.readBlock(vals.data(), 0, 0, width, length);

    std::vector<int> expected = {  0,  1,  2,
                                   3,  4,  5,
                                   6,  7,  8,
                                   9, 10, 11 };

    EXPECT_EQ( vals, expected );
}

TEST_F(MEMRasterTest, WriteBlockColMajor)
{
    std::size_t rowstride = sizeof(int);
    std::size_t colstride = length * rowstride;
    Raster raster(v.data(), width, length, colstride, rowstride);

    std::vector<int> vals = {  0,  1,  2,
                               3,  4,  5,
                               6,  7,  8,
                               9, 10, 11 };

    raster.writeBlock(vals.data(), 0, 0, width, length);

    std::vector<int> expected = {  0,  3,  6,  9,
                                   1,  4,  7, 10,
                                   2,  5,  8, 11 };

    EXPECT_EQ( v, expected );

    // writing to read-only raster should throw
    {
        Raster raster(v.data(), width, length, colstride, rowstride, GA_ReadOnly);
        EXPECT_THROW( { raster.writeBlock(vals.data(), 0, 0, width, length); }, isce3::except::RuntimeError );
    }
}

TEST_F(MEMRasterTest, Memmap)
{
    v = { 0,  1,  2,
          3,  4,  5,
          6,  7,  8,
          9, 10, 11 };

    Raster raster(v.data(), width, length);

    {
        Buffer mmap = raster.memmap();

        int * data = reinterpret_cast<int *>(mmap.data());
        for (int i = 0; i < raster.length() * raster.width(); ++i) {
            EXPECT_EQ( data[i], i );
        }

        EXPECT_EQ( mmap.datatype(), raster.datatype() );
        EXPECT_EQ( mmap.length(), raster.length() );
        EXPECT_EQ( mmap.width(), raster.width() );
        EXPECT_EQ( mmap.colstride(), sizeof(int) );
        EXPECT_EQ( mmap.rowstride(), raster.width() * sizeof(int) );
    }

    {
        TypedBuffer<int> mmap = raster.memmap<int>();

        int * data = mmap.data();
        for (int i = 0; i < raster.length() * raster.width(); ++i) {
            EXPECT_EQ( data[i], i );
        }

        EXPECT_EQ( mmap.datatype(), raster.datatype() );
        EXPECT_EQ( mmap.length(), raster.length() );
        EXPECT_EQ( mmap.width(), raster.width() );
        EXPECT_EQ( mmap.colstride(), sizeof(int) );
        EXPECT_EQ( mmap.rowstride(), raster.width() * sizeof(int) );
    }
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
