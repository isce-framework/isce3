#include <array>
#include <gtest/gtest.h>

#include <isce3/except/Error.h>
#include <isce3/io/gdal/GeoTransform.h>

using isce::io::gdal::GeoTransform;

TEST(GeoTransformTest, DefaultConstructor)
{
    GeoTransform transform;

    EXPECT_EQ( transform.x0, 0. );
    EXPECT_EQ( transform.y0, 0. );
    EXPECT_EQ( transform.dx, 1. );
    EXPECT_EQ( transform.dy, 1. );
}

TEST(GeoTransformTest, ValuesConstructor)
{
    double x0 = 1.;
    double y0 = 2.;
    double dx = 3.;
    double dy = 4.;
    GeoTransform transform(x0, y0, dx, dy);

    EXPECT_EQ( transform.x0, x0 );
    EXPECT_EQ( transform.y0, y0 );
    EXPECT_EQ( transform.dx, dx );
    EXPECT_EQ( transform.dy, dy );
}

TEST(GeoTransformTest, AffineCoeffsConstructor)
{
    std::array<double, 6> coeffs = { 1., 2., 0., 3., 0., 4. };
    GeoTransform transform(coeffs);

    EXPECT_EQ( transform.x0, coeffs[0] );
    EXPECT_EQ( transform.y0, coeffs[3] );
    EXPECT_EQ( transform.dx, coeffs[1] );
    EXPECT_EQ( transform.dy, coeffs[5] );
}

TEST(GeoTransformTest, InvalidGeoTransform)
{
    std::array<double, 6> coeffs = { 1., 2., 3., 4., 5., 6. };
    EXPECT_THROW( { GeoTransform transform(coeffs); }, isce::except::InvalidArgument );
}

TEST(GeoTransformTest, GetCoeffs)
{
    std::array<double, 6> coeffs = { 1., 2., 0., 3., 0., 4. };
    GeoTransform transform(coeffs);

    EXPECT_EQ( transform.getCoeffs(), coeffs );
}

TEST(GeoTransformTest, IsIdentity)
{
    {
        GeoTransform transform;
        EXPECT_TRUE( transform.isIdentity() );
    }

    {
        GeoTransform transform(1., 2., 3., 4.);
        EXPECT_FALSE( transform.isIdentity() );
    }
}

TEST(GeoTransformTest, Comparison)
{
    GeoTransform transform1;
    GeoTransform transform2;
    GeoTransform transform3(1., 2., 3., 4.);

    EXPECT_TRUE( transform1 == transform2 );
    EXPECT_TRUE( transform1 != transform3 );
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
