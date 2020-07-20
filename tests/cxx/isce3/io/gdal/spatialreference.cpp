#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include <isce3/except/Error.h>
#include <isce3/io/gdal/SpatialReference.h>

using isce::io::gdal::SpatialReference;

struct SpatialReferenceTest : public testing::TestWithParam<int> {};

TEST_P(SpatialReferenceTest, EPSG)
{
    int epsg = GetParam();
    SpatialReference srs(epsg);

    EXPECT_EQ( srs.toEPSG(), epsg );
}

INSTANTIATE_TEST_SUITE_P(EPSG_4326, SpatialReferenceTest, testing::Values(4326));
INSTANTIATE_TEST_SUITE_P(EPSG_3031, SpatialReferenceTest, testing::Values(3031));
INSTANTIATE_TEST_SUITE_P(EPSG_3413, SpatialReferenceTest, testing::Values(3413));
INSTANTIATE_TEST_SUITE_P(EPSG_6933, SpatialReferenceTest, testing::Values(6933));
INSTANTIATE_TEST_SUITE_P(UTM_North, SpatialReferenceTest, testing::Range(32601, 32661));
INSTANTIATE_TEST_SUITE_P(UTM_South, SpatialReferenceTest, testing::Range(32701, 32761));

TEST(SpatialReferenceTest, InvalidEPSG)
{
    EXPECT_THROW( { SpatialReference srs(0); }, isce::except::GDALError );
}

TEST(SpatialReferenceTest, WKT)
{
    std::string wkt =
        "GEOGCS[\"WGS 84\","
            "DATUM[\"WGS_1984\","
                "SPHEROID[\"WGS 84\",6378137,298.257223563,"
                    "AUTHORITY[\"EPSG\",\"7030\"]],"
                "AUTHORITY[\"EPSG\",\"6326\"]],"
            "PRIMEM[\"Greenwich\",0,"
                "AUTHORITY[\"EPSG\",\"8901\"]],"
            "UNIT[\"degree\",0.0174532925199433,"
                "AUTHORITY[\"EPSG\",\"9122\"]],"
            "AUTHORITY[\"EPSG\",\"4326\"]]";

    SpatialReference srs1(wkt);
    std::string wkt_out = srs1.toWKT();
    std::cout << wkt_out << std::endl;

    SpatialReference srs2(wkt_out);
    EXPECT_EQ( srs1, srs2 );
}

TEST(SpatialReferenceTest, Comparison)
{
    SpatialReference srs1(4326);
    SpatialReference srs2(4326);
    SpatialReference srs3(6933);

    EXPECT_TRUE( srs1 == srs2 );
    EXPECT_TRUE( srs1 != srs3 );
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
