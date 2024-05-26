#include <thrust/device_vector.h>
#include <vector>

#include <gtest/gtest.h>

#include <isce3/cuda/geocode/MaskedMinMax.h>

struct MaskedMinMaxTest : public ::testing::Test {
    std::vector<double> data = {5.0, 2.0, 4.0, 3.0, 1.0};
};

TEST_F(MaskedMinMaxTest, NoMaskedData)
{
    // copy common data to device
    thrust::device_vector<double> d_data(data);

    // create mask with nothing masked
    std::vector<bool> mask = {false, false, false, false, false};

    // copy to device
    thrust::device_vector<bool> d_mask(mask);

    const auto [data_min, data_max] = isce3::cuda::geocode::masked_minmax(d_data, d_mask);

    ASSERT_EQ(data_min, 1.0);
    ASSERT_EQ(data_max, 5.0);
}

TEST_F(MaskedMinMaxTest, SomeMaskedData)
{
    // copy common data to device
    thrust::device_vector<double> d_data(data);

    // create mask with head and tail masked
    std::vector<bool> mask = {true, false, false, false, true};
    thrust::device_vector<bool> d_mask(mask);

    const auto [data_min, data_max] = isce3::cuda::geocode::masked_minmax(d_data, d_mask);

    ASSERT_EQ(data_min, 2.0);
    ASSERT_EQ(data_max, 4.0);
}

TEST_F(MaskedMinMaxTest, AllMaskedData)
{
    // copy to common data device
    thrust::device_vector<double> d_data(data);

    // create mask with everything masked
    std::vector<bool> mask = {true, true, true, true, true};

    // copy to device
    thrust::device_vector<bool> d_mask(mask);

    const auto [data_min, data_max] = isce3::cuda::geocode::masked_minmax(d_data, d_mask);

    ASSERT_TRUE(isnan(data_min));
    ASSERT_TRUE(isnan(data_max));
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
