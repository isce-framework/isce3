#include <sstream>

#include <gtest/gtest.h>

#include <isce3/cuda/core/ComputeCapability.h>

using isce3::cuda::core::ComputeCapability;

TEST(ComputeCapabilityTest, Constructor)
{
    const ComputeCapability compute(2, 0);
    EXPECT_EQ(compute.major, 2);
    EXPECT_EQ(compute.minor, 0);
}

TEST(ComputeCapabilityTest, ToString)
{
    const ComputeCapability compute(3, 5);
    EXPECT_EQ(std::string(compute), "3.5");
}

TEST(ComputeCapabilityTest, SerializeToStream)
{
    const ComputeCapability compute(3, 5);
    std::ostringstream ss;
    ss << compute;
    EXPECT_EQ(ss.str(), "3.5");
}

TEST(ComputeCapabilityTest, Comparison)
{
    const ComputeCapability compute1(3, 2);
    const ComputeCapability compute2(3, 2);
    const ComputeCapability compute3(3, 5);
    const ComputeCapability compute4(5, 0);

    EXPECT_TRUE(compute1 == compute2);
    EXPECT_TRUE(compute1 != compute3);
    EXPECT_TRUE(compute1 < compute4);
    EXPECT_TRUE(compute3 > compute1);
    EXPECT_TRUE(compute2 <= compute1);
    EXPECT_TRUE(compute4 >= compute3);
}

TEST(ComputeCapabilityTest, MinComputeCapability)
{
    const auto min_compute = isce3::cuda::core::minComputeCapability();
    EXPECT_GE(min_compute.major, 1);
    EXPECT_GE(min_compute.minor, 0);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
