#include <gtest/gtest.h>
#include <isce3/focus/GapMask.h>

TEST(GapDetectionTest, Mask)
{
    // Send a 1 s pulse every 2 s so that half of samples are blocked.
    // With fs = 1 and dwp = 10 then it's even samples.
    int m = 100;
    std::vector<double> t(m);
    double pri = 2.0;
    for (int i = 0; i < m; ++i) {
        t[i] = i * pri;
    }
    int n = 10;
    double fs = 1.0;
    double dwp = 10.0;
    double chirplen = 1.0;
    isce::focus::GapMask masker(t, n, dwp, fs, chirplen);
    auto mask = masker.mask(0);
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(i % 2 == 0, mask[i]);
    }
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
