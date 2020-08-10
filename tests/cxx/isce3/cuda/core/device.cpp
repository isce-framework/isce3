#include <gtest/gtest.h>

#include <isce3/cuda/core/Device.h>
#include <isce3/except/Error.h>

using isce3::cuda::core::Device;
using isce3::cuda::core::getDevice;
using isce3::cuda::core::getDeviceCount;
using isce3::cuda::core::setDevice;
using isce3::except::InvalidArgument;

struct DeviceTest : public testing::Test {

    void SetUp() override
    {
        count = getDeviceCount();
        EXPECT_GE(count, 1);
    }

    int count = 0;
};

TEST_F(DeviceTest, Device)
{
    for (int id = 0; id < count; ++id) {
        const Device device(id);
        EXPECT_EQ(device.id(), id);

        std::cout << "Device " << device.id() << std::endl
                  << "--------" << std::endl
                  << "name: " << device.name() << std::endl
                  << "compute: " << device.computeCapability() << std::endl
                  << "total mem (bytes): " << device.totalGlobalMem() << std::endl;

        EXPECT_NE(device.name(), "");
        EXPECT_GT(device.totalGlobalMem(), 0);

        const auto compute = device.computeCapability();
        EXPECT_GE(compute.major, 1);
        EXPECT_GE(compute.minor, 0);
    }
}

TEST_F(DeviceTest, InvalidDevice)
{
    EXPECT_THROW({ const Device device(-1); }, InvalidArgument);
    EXPECT_THROW({ const Device device(count); }, InvalidArgument);
}

TEST_F(DeviceTest, GetDevice)
{
    const Device device = getDevice();
    EXPECT_GE(device.id(), 0);
}

TEST_F(DeviceTest, SetDevice)
{
    for (int id = 0; id < count; ++id) {
        const Device device(id);
        setDevice(id);
        EXPECT_EQ(getDevice(), device);
    }
}

TEST_F(DeviceTest, Comparison)
{
    const Device device1(0);
    const Device device2(0);
    EXPECT_TRUE(device1 == device2);

    if (count > 1) {
        const Device device3(1);
        EXPECT_TRUE(device1 != device3);
    }
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
