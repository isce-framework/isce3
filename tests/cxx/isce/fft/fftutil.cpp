#include <cstdint>
#include <gtest/gtest.h>

#include <isce/except/Error.h>
#include <isce/fft/FFTUtil.h>

using isce::fft::nextPowerOfTwo;
using isce::fft::nextFastPower;

TEST(FFTUtilTest, NextPowerOfTwo)
{
    EXPECT_THROW( { nextPowerOfTwo(-1); }, isce::except::DomainError );

    EXPECT_EQ( nextPowerOfTwo(0), 1 );
    EXPECT_EQ( nextPowerOfTwo(1), 1 );
    EXPECT_EQ( nextPowerOfTwo(19), 32 );
    EXPECT_EQ( nextPowerOfTwo(256), 256 );
    EXPECT_EQ( nextPowerOfTwo(257), 512 );

    auto n = std::uint64_t(1) << 50;
    EXPECT_EQ( nextPowerOfTwo(n-1), n );
}

TEST(FFTUtilTest, NextFastPower)
{
    EXPECT_THROW( { nextFastPower(-1); }, isce::except::DomainError );

    EXPECT_EQ( nextFastPower(0), 1 );
    EXPECT_EQ( nextFastPower(1), 1 );
    EXPECT_EQ( nextFastPower(19), 20 );
    EXPECT_EQ( nextFastPower(256), 256 );
    EXPECT_EQ( nextFastPower(257), 270 );
    // bigger than sqrt(INT_MAX)
    EXPECT_EQ( nextFastPower(1<<18), 1<<18 );
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
