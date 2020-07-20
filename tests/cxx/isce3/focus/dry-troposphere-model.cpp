#include <gtest/gtest.h>
#include <isce3/except/Error.h>
#include <isce3/focus/DryTroposphereModel.h>

using isce::except::InvalidArgument;
using isce::focus::DryTroposphereModel;
using isce::focus::parseDryTropoModel;
using isce::focus::toString;

TEST(DryTroposphereModelTest, ToString)
{
    EXPECT_EQ(toString(DryTroposphereModel::NoDelay), "nodelay");
    EXPECT_EQ(toString(DryTroposphereModel::TSX), "tsx");
}

TEST(DryTroposphereModelTest, FromString)
{
    EXPECT_EQ(parseDryTropoModel("nodelay"), DryTroposphereModel::NoDelay);
    EXPECT_EQ(parseDryTropoModel("tsx"), DryTroposphereModel::TSX);

    EXPECT_THROW({parseDryTropoModel("asdf");}, InvalidArgument);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
