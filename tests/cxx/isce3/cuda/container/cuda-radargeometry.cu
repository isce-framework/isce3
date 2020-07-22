#include <gtest/gtest.h>

#include <isce3/core/Serialization.h>
#include <isce3/cuda/container/RadarGeometry.h>
#include <isce3/io/IH5.h>

using HostRadarGeometry = isce3::container::RadarGeometry;
using DeviceRadarGeometry = isce3::cuda::container::RadarGeometry;

struct RadarGeometryTest : public testing::Test {

    isce3::product::RadarGridParameters radar_grid;
    isce3::core::Orbit orbit;
    isce3::core::LUT2d<double> doppler;

    void SetUp() override
    {
        std::string filename = TESTDATA_DIR "point-target-sim-rc.h5";
        auto h5file = isce3::io::IH5File(filename, 'r');

        // load orbit
        auto orbit_grp = h5file.openGroup("orbit");
        isce3::core::loadFromH5(orbit_grp, orbit);

        // load radar grid parameters
        double sensing_start_time;
        h5file.openDataSet("time_of_first_pulse").read(&sensing_start_time);

        double azimuth_spacing;
        h5file.openDataSet("pulse_spacing").read(&azimuth_spacing);

        double two_way_range_delay;
        h5file.openDataSet("two_way_range_delay").read(&two_way_range_delay);

        double range_sampling_rate;
        h5file.openDataSet("range_sample_rate").read(&range_sampling_rate);

        double centerfreq;
        h5file.openDataSet("center_frequency").read(&centerfreq);

        static constexpr double c = isce3::core::speed_of_light;
        double near_range = c / 2. * two_way_range_delay;
        double range_spacing = c / (2. * range_sampling_rate);
        double wavelength = c / centerfreq;

        std::string look_side_str;
        h5file.openDataSet("look_side").read(look_side_str);
        auto look_side = isce3::core::parseLookSide(look_side_str);

        auto data_ds = h5file.openDataSet("data");
        auto shape = data_ds.getDimensions();
        size_t lines = shape[0];
        size_t samples = shape[1];

        const double prf = 1. / azimuth_spacing;

        radar_grid = isce3::product::RadarGridParameters(
                sensing_start_time, wavelength, prf, near_range, range_spacing,
                look_side, lines, samples, orbit.referenceEpoch());

        // load Doppler
        auto doppler_grp = h5file.openGroup("doppler");
        isce3::core::loadCalGrid(doppler_grp, "doppler", doppler);
    }
};

TEST_F(RadarGeometryTest, FromHost)
{
    auto h_rdrgeom = HostRadarGeometry(radar_grid, orbit, doppler);
    auto d_rdrgeom = DeviceRadarGeometry(h_rdrgeom);

    EXPECT_EQ(d_rdrgeom.referenceEpoch(), h_rdrgeom.referenceEpoch());
    EXPECT_EQ(d_rdrgeom.gridLength(), h_rdrgeom.gridLength());
    EXPECT_EQ(d_rdrgeom.gridWidth(), h_rdrgeom.gridWidth());
    EXPECT_EQ(d_rdrgeom.lookSide(), h_rdrgeom.lookSide());

    EXPECT_EQ(d_rdrgeom.radarGrid().length(), radar_grid.length());
    EXPECT_EQ(d_rdrgeom.radarGrid().width(), radar_grid.width());
    EXPECT_DOUBLE_EQ(d_rdrgeom.radarGrid().wavelength(),
                     radar_grid.wavelength());

    EXPECT_EQ(d_rdrgeom.orbit().size(), orbit.size());
    EXPECT_DOUBLE_EQ(d_rdrgeom.orbit().spacing(), orbit.spacing());

    EXPECT_EQ(d_rdrgeom.doppler().length(), doppler.length());
    EXPECT_EQ(d_rdrgeom.doppler().width(), doppler.width());
    EXPECT_DOUBLE_EQ(d_rdrgeom.doppler().xStart(), doppler.xStart());
    EXPECT_DOUBLE_EQ(d_rdrgeom.doppler().ySpacing(), doppler.ySpacing());
}

TEST_F(RadarGeometryTest, SensingTime)
{
    auto rdrgeom = DeviceRadarGeometry(radar_grid, orbit, doppler);
    auto sensing_time = rdrgeom.sensingTime();

    EXPECT_DOUBLE_EQ(sensing_time[0], radar_grid.sensingStart());
    EXPECT_DOUBLE_EQ(sensing_time.spacing(), radar_grid.azimuthTimeInterval());
    EXPECT_EQ(sensing_time.size(), radar_grid.length());
}

TEST_F(RadarGeometryTest, SlantRange)
{
    auto rdrgeom = DeviceRadarGeometry(radar_grid, orbit, doppler);
    auto slant_range = rdrgeom.slantRange();

    EXPECT_DOUBLE_EQ(slant_range[0], radar_grid.startingRange());
    EXPECT_DOUBLE_EQ(slant_range.spacing(), radar_grid.rangePixelSpacing());
    EXPECT_EQ(slant_range.size(), radar_grid.width());
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
