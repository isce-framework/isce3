#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

#include <isce/core/Kernels.h>
#include <isce/focus/Chirp.h>
#include <isce/focus/RangeComp.h>

using isce::core::sinc;
using isce::focus::formLinearChirp;
using isce::focus::RangeComp;

template<typename T>
auto maxAbsError(const std::vector<T>& a, const std::vector<T>& b)
{
    if (a.size() != b.size()) {
        throw std::length_error("inputs must be the same size");
    }
    std::size_t size = a.size();

    using U = decltype(std::abs(b[0] - a[0]));
    U mae = 0.;
    for (std::size_t i = 0; i < size; ++i) {
        mae = std::max(mae, std::abs(b[i] - a[i]));
    }

    return mae;
}

TEST(RangeCompTest, Constructor)
{
    double chirprate = 1e12;
    double duration = 20e-6;
    double samplerate = 24e6;
    std::vector<std::complex<float>> chirp = formLinearChirp(chirprate, duration, samplerate);

    int inputsize = 1000;
    int maxbatch = 8;

    // mode == full
    {
        RangeComp::Mode mode = RangeComp::Mode::Full;
        int outputsize = inputsize + chirp.size() - 1;

        RangeComp rcproc(chirp, inputsize, maxbatch, mode);

        EXPECT_EQ(rcproc.chirpSize(), chirp.size());
        EXPECT_EQ(rcproc.inputSize(), inputsize);
        EXPECT_GE(rcproc.fftSize(), inputsize + chirp.size() - 1);
        EXPECT_EQ(rcproc.maxBatch(), maxbatch);
        EXPECT_EQ(rcproc.outputSize(), outputsize);
        EXPECT_EQ(rcproc.mode(), mode);
    }

    // mode == valid
    {
        RangeComp::Mode mode = RangeComp::Mode::Valid;
        int outputsize = inputsize - chirp.size() + 1;

        RangeComp rcproc(chirp, inputsize, maxbatch, mode);

        EXPECT_EQ(rcproc.chirpSize(), chirp.size());
        EXPECT_EQ(rcproc.inputSize(), inputsize);
        EXPECT_GE(rcproc.fftSize(), inputsize + chirp.size() - 1);
        EXPECT_EQ(rcproc.maxBatch(), maxbatch);
        EXPECT_EQ(rcproc.outputSize(), outputsize);
        EXPECT_EQ(rcproc.mode(), mode);
    }

    // mode == same
    {
        RangeComp::Mode mode = RangeComp::Mode::Same;

        RangeComp rcproc(chirp, inputsize, maxbatch, mode);

        EXPECT_EQ(rcproc.chirpSize(), chirp.size());
        EXPECT_EQ(rcproc.inputSize(), inputsize);
        EXPECT_GE(rcproc.fftSize(), inputsize + chirp.size() - 1);
        EXPECT_EQ(rcproc.maxBatch(), maxbatch);
        EXPECT_EQ(rcproc.outputSize(), inputsize);
        EXPECT_EQ(rcproc.mode(), mode);
    }
}

TEST(RangeCompTest, RangeCompress)
{
    // This test performs range compression on a signal which is an exact replica
    // of the template chirp and compares the output to the analytical result.
    // There's some discrepancy between the results for discrete and continuous
    // signals, which, for purposes of testing, are minimized by highly oversampling
    // the chirp and comparing the results only around the first few central lobes,
    // where the error is smallest.

    double chirprate = 100.;
    double duration = 2.;
    double samplerate = 2400.;
    std::vector<std::complex<float>> chirp = formLinearChirp(chirprate, duration, samplerate);

    // signal is an exact replica of chirp waveform
    RangeComp rcproc(chirp, static_cast<int>(chirp.size()));
    std::vector<std::complex<float>> output(rcproc.outputSize());
    rcproc.rangecompress(output.data(), chirp.data());

    // in order to be consistent with the result for a continuous signal, we
    // need to multiply by the sample spacing
    double spacing = 1. / samplerate;
    for (auto & z : output) { z *= spacing; }

    // compute the analytical result
    // Ref: Cumming, Wong, ‘Digital Processing of Synthetic Aperture Radar
    //      Data: Algorithms and Implementation’, 2004
    std::vector<std::complex<float>> expected(2 * chirp.size() - 1);
    double T = chirp.size() / samplerate;
    double t0 = 0.5 * (expected.size() - 1) / samplerate;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        double t = i / samplerate - t0;
        expected[i] = (T - std::abs(t)) * sinc(chirprate * t * (T - std::abs(t)));
    }

    // take a centered slice of the results and compare their values
    int n = rcproc.outputSize();
    int k = 200;
    std::vector<std::complex<float>> output_slice(&output[n/2 - k], &output[n/2 + k+1]);
    std::vector<std::complex<float>> expected_slice(&expected[n/2 - k], &expected[n/2 + k+1]);
    float mae = maxAbsError(output_slice, expected_slice);
    float errtol = 1e-6;
    EXPECT_LT(mae, errtol);
}

TEST(RangeCompTest, ConvolveMode)
{
    int chirpsize = 5;
    int inputsize = 9;
    std::vector<std::complex<float>> chirp(chirpsize, 1.);
    std::vector<std::complex<float>> input(inputsize, 1.);

    float errtol = 1e-6;

    // mode == full
    {
        RangeComp::Mode mode = RangeComp::Mode::Full;
        RangeComp rcproc(chirp, inputsize, 1, mode);

        std::vector<std::complex<float>> output(rcproc.outputSize());
        rcproc.rangecompress(output.data(), input.data());

        std::vector<std::complex<float>> expected = {1., 2., 3., 4., 5., 5., 5., 5., 5., 4., 3., 2., 1.};
        float mae = maxAbsError(output, expected);
        EXPECT_LT(mae, errtol);

        EXPECT_EQ(rcproc.firstValidSample(), 4);
    }

    // mode == valid
    {
        RangeComp::Mode mode = RangeComp::Mode::Valid;
        RangeComp rcproc(chirp, inputsize, 1, mode);

        std::vector<std::complex<float>> output(rcproc.outputSize());
        rcproc.rangecompress(output.data(), input.data());

        std::vector<std::complex<float>> expected = {5., 5., 5., 5., 5.};
        float mae = maxAbsError(output, expected);
        EXPECT_LT(mae, errtol);

        EXPECT_EQ(rcproc.firstValidSample(), 0);
    }

    // mode == same
    {
        RangeComp::Mode mode = RangeComp::Mode::Same;
        RangeComp rcproc(chirp, inputsize, 1, mode);

        std::vector<std::complex<float>> output(rcproc.outputSize());
        rcproc.rangecompress(output.data(), input.data());

        std::vector<std::complex<float>> expected = {3., 4., 5., 5., 5., 5., 5., 4., 3.};
        float mae = maxAbsError(output, expected);
        EXPECT_LT(mae, errtol);

        EXPECT_EQ(rcproc.firstValidSample(), 2);
    }
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
