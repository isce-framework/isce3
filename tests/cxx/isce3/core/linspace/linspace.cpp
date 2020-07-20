#include <gtest/gtest.h>

#include <isce3/core/Linspace.h>

TEST(LinspaceTest, BasicConstructor)
{
    int first = 0;
    int spacing = 1;
    int size = 10;

    isce3::core::Linspace<int> x(first, spacing, size);

    EXPECT_EQ( x.first(), first );
    EXPECT_EQ( x.spacing(), spacing );
    EXPECT_EQ( x.size(), size );
}

TEST(LinspaceTest, FromInterval)
{
    int first = 0;
    int spacing = 1;
    int size = 10;

    int last = first + spacing * (size - 1);

    typedef isce3::core::Linspace<int> Linspace;

    Linspace x = Linspace::from_interval(first, last, size);

    EXPECT_EQ( x.first(), first );
    EXPECT_EQ( x.last(), last );
    EXPECT_EQ( x.spacing(), spacing );
    EXPECT_EQ( x.size(), size );
}

TEST(LinspaceTest, CopyConstructor)
{
    float first = 0.;
    float spacing = 1.;
    int size = 10;

    isce3::core::Linspace<float> x1(first, spacing, size);
    isce3::core::Linspace<double> x2(x1);

    EXPECT_FLOAT_EQ( x2.first(), x1.first() );
    EXPECT_FLOAT_EQ( x2.spacing(), x1.spacing() );
    EXPECT_EQ( x2.size(), x1.size() );
}

TEST(LinspaceTest, Assignment)
{
    float first = 0.;
    float spacing = 1.;
    int size = 10;

    isce3::core::Linspace<float> x1(first, spacing, size);
    isce3::core::Linspace<double> x2;
    x2 = x1;

    EXPECT_FLOAT_EQ( x2.first(), x1.first() );
    EXPECT_FLOAT_EQ( x2.spacing(), x1.spacing() );
    EXPECT_EQ( x2.size(), x1.size() );
}

TEST(LinspaceTest, Accessor)
{
    int first = 0;
    int spacing = 1;
    int size = 10;

    isce3::core::Linspace<int> x(first, spacing, size);

    EXPECT_EQ( x[0], first );
    EXPECT_EQ( x[1], first + spacing );
    EXPECT_EQ( x[5], first + 5 * spacing );
}

TEST(LinspaceTest, SetFirst)
{
    int first = 0;
    int spacing = 1;
    int size = 10;

    isce3::core::Linspace<int> x(first, spacing, size);

    int new_first = 10;

    x.first(new_first);

    EXPECT_EQ( x.first(), new_first );
    EXPECT_EQ( x[0], new_first );
    EXPECT_EQ( x[1], new_first + spacing );
    EXPECT_EQ( x[5], new_first + 5 * spacing );
}

TEST(LinspaceTest, SetSpacing)
{
    int first = 0;
    int spacing = 1;
    int size = 10;

    isce3::core::Linspace<int> x(first, spacing, size);

    int new_spacing = 2;

    x.spacing(new_spacing);

    EXPECT_EQ( x.spacing(), new_spacing );
    EXPECT_EQ( x[1] - x[0], new_spacing );
}

TEST(LinspaceTest, Resize)
{
    int first = 0;
    int spacing = 1;
    int size = 10;

    isce3::core::Linspace<int> x(first, spacing, size);

    int new_size = 15;

    x.resize(new_size);

    EXPECT_EQ( x.size(), new_size );
}

TEST(LinspaceTest, SubInterval)
{
    int first = 0;
    int spacing = 1;
    int size = 10;

    isce3::core::Linspace<int> x1(first, spacing, size);

    int start = 3;
    int stop = 8;

    isce3::core::Linspace<int> x2 = x1.subinterval(start, stop);

    EXPECT_EQ( x2.first(), first + start * spacing );
    EXPECT_EQ( x2.last(), first + (stop - 1) * spacing );
    EXPECT_EQ( x2.spacing(), spacing );
    EXPECT_EQ( x2.size(), stop - start );
}

TEST(LinspaceTest, Empty)
{
    int first = 0;
    int spacing = 1;

    isce3::core::Linspace<int> x1(first, spacing, 0);
    isce3::core::Linspace<int> x2(first, spacing, 10);

    EXPECT_TRUE( x1.empty() );
    EXPECT_FALSE( x2.empty() );
}

TEST(LinspaceTest, Search)
{
    // sample spacing > 0
    {
        int first = 0;
        int spacing = 1;
        int size = 10;

        isce3::core::Linspace<int> x(first, spacing, size);

        EXPECT_EQ( x.search(-1), 0 );
        EXPECT_EQ( x.search(2.5), 3 );
        EXPECT_EQ( x.search(11), 10 );
    }

    // sample spacing < 0
    {
        int first = 0;
        int spacing = -1;
        int size = 10;

        isce3::core::Linspace<int> x(first, spacing, size);

        EXPECT_EQ( x.search(1), 0 );
        EXPECT_EQ( x.search(-2.5), 3 );
        EXPECT_EQ( x.search(-11), 10 );
    }
}

TEST(LinspaceTest, Comparison)
{
    int first = 0;
    int spacing = 1;
    int size = 10;

    isce3::core::Linspace<int> x1(first, spacing, size);
    isce3::core::Linspace<int> x2(first, spacing, size);
    isce3::core::Linspace<int> x3(first + 1, spacing, size);

    EXPECT_TRUE( x1 == x2 );
    EXPECT_TRUE( x1 != x3 );
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
