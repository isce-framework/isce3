#include <gtest/gtest.h>

#include <isce3/container/RSD.h>

using isce3::container::RSD;
using isce3::core::DateTime;
using isce3::core::LookSide;
using isce3::io::gdal::Raster;

struct RSDTest : public testing::Test {
    int lines = 21;
    int samples = 11;

    DateTime reference_epoch;

    double azimuth_spacing = 0.0005;
    double azimuth_start_time = 0.;
    double azimuth_mid_time = 0.005;
    double azimuth_end_time = 0.01;

    double range_sampling_rate = 50e6;
    double range_window_start_time = 0.005;
    double range_window_mid_time = 0.0050001;
    double range_window_end_time = 0.0050002;

    LookSide look_side = LookSide::Left;

    std::vector<double> azimuth_time;

    std::vector<std::complex<float>> iq;

    void SetUp() override
    {
        reference_epoch = DateTime(2000, 1, 1);

        azimuth_time.resize(lines);
        for (int l = 0; l < lines; ++l) {
            azimuth_time[l] = azimuth_start_time + l * azimuth_spacing;
        }

        iq.resize(lines * samples);
        for (int i = 0; i < lines * samples; ++i) {
            iq[i] = {float(i), float(-i)};
        }
    }
};

TEST_F(RSDTest, RSD)
{
    Raster signal_data(iq.data(), samples, lines);

    RSD rsd(signal_data, reference_epoch, azimuth_time,
            range_window_start_time, range_sampling_rate,
            look_side);

    EXPECT_EQ( rsd.lines(), lines );
    EXPECT_EQ( rsd.samples(), samples );
    EXPECT_EQ( rsd.referenceEpoch(), reference_epoch );
    EXPECT_DOUBLE_EQ( rsd.azimuthStartTime(), azimuth_start_time );
    EXPECT_DOUBLE_EQ( rsd.azimuthMidTime(), azimuth_mid_time );
    EXPECT_DOUBLE_EQ( rsd.azimuthEndTime(), azimuth_end_time );
    EXPECT_DOUBLE_EQ( rsd.rangeWindowStartTime(), range_window_start_time );
    EXPECT_DOUBLE_EQ( rsd.rangeWindowMidTime(), range_window_mid_time );
    EXPECT_DOUBLE_EQ( rsd.rangeWindowEndTime(), range_window_end_time );
    EXPECT_DOUBLE_EQ( rsd.rangeSamplingRate(), range_sampling_rate );
    EXPECT_EQ( rsd.lookSide(), look_side );
}

TEST_F(RSDTest, ReadLine)
{
    Raster signal_data(iq.data(), samples, lines);

    RSD rsd(signal_data, reference_epoch, azimuth_time,
            range_window_start_time, range_sampling_rate,
            look_side);

    int line = 5;
    std::vector<std::complex<float>> out(samples);
    rsd.readLine(out.data(), line);

    std::vector<std::complex<float>> expected(&iq[line * samples],
                                              &iq[(line + 1) * samples]);

    EXPECT_EQ( out, expected );
}

TEST_F(RSDTest, ReadLines)
{
    Raster signal_data(iq.data(), samples, lines);

    RSD rsd(signal_data, reference_epoch, azimuth_time,
            range_window_start_time, range_sampling_rate,
            look_side);

    int first_line = 3;
    int num_lines = 4;
    std::vector<std::complex<float>> out(num_lines * samples);
    rsd.readLines(out.data(), first_line, num_lines);

    std::vector<std::complex<float>> expected(&iq[first_line * samples],
                                              &iq[(first_line + num_lines) * samples]);

    EXPECT_EQ( out, expected );
}

TEST_F(RSDTest, ReadAll)
{
    Raster signal_data(iq.data(), samples, lines);

    RSD rsd(signal_data, reference_epoch, azimuth_time,
            range_window_start_time, range_sampling_rate,
            look_side);

    std::vector<std::complex<float>> out(lines * samples);
    rsd.readAll(out.data());

    EXPECT_EQ( out, iq );
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
