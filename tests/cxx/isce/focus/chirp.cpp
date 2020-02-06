#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

#include <isce/focus/Chirp.h>

using isce::focus::formLinearChirp;

TEST(FormLinearChirpTest, FormLinearChirp)
{
    double startfreq = 0.;
    double endfreq = 20.;
    double duration = 1.;
    int samples = 1001;
    double amplitude = 7.5;
    double phi = 0.5 * M_PI;

    double centerfreq = 0.5 * (startfreq + endfreq);
    double chirprate = (endfreq - startfreq) / duration;
    double spacing = duration / (samples - 1);
    double samplerate = 1. / spacing;

    std::vector<std::complex<float>> chirp = formLinearChirp(
            chirprate, duration, samplerate, centerfreq, amplitude, phi);

    // check number of samples
    EXPECT_EQ( chirp.size(), samples );

    // check amplitudes
    {
        double maxerr = 0.;
        for (int i = 0; i < samples; ++i) {
            maxerr = std::max(maxerr, std::abs(amplitude - std::abs(chirp[i])));
        }

        EXPECT_LT(maxerr, 1e-6);
    }

    // check initial phase
    EXPECT_FLOAT_EQ( std::arg(chirp[0]), phi );

    // check phase difference between each adjacent pair of samples
    {
        double maxerr = 0.;
        for (int i = 0; i < samples - 1; ++i) {

            // instantaneous frequency at the center of the two samples
            double f = startfreq + (endfreq - startfreq) * (i + 0.5) / samples;

            std::complex<double> z1 = chirp[i];
            std::complex<double> z2 = chirp[i + 1];
            double dphi = std::arg(z2 * std::conj(z1));

            maxerr = std::max(maxerr, std::abs(2. * M_PI * f * spacing - dphi));
        }

        EXPECT_LT(maxerr, 1e-3);
    }
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
