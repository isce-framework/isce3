#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include <isce3/core/EMatrix.h>
#include <isce3/signal/flatten.h>
#include <isce3/signal/CrossMultiply.h>

TEST(CrossMultiply, RunCrossMultiply)
{

    int length = 100;
    int width = 50;
    int upsample = 2;

    isce3::core::EArray2D<std::complex<float>> ref_slc(length, width);
    isce3::core::EArray2D<std::complex<float>> sec_slc(length, width);
    isce3::core::EArray2D<std::complex<float>> ifgram(length, width);
    isce3::core::EArray2D<std::complex<float>> expected_ifgram(length, width);

    for (int i = 0; i < length; i++) {
        for (int j = 0; j < width; j++) {
            double phase = std::sin(10 * M_PI * i / width);
            double phase2 = std::sin(8 * M_PI * i / width);

            ref_slc(i, j) = std::polar(1., phase);
            sec_slc(i, j) = std::polar(1., phase2);
            double r1 = ref_slc(i, j).real();
            double im1 = ref_slc(i, j).imag();
            double r2 = sec_slc(i, j).real();
            double im2 = sec_slc(i, j).imag();

            double r = r1 * r2 + im1 * im2;
            double im = im1 * r2 - r1 * im2;
            expected_ifgram(i, j) = std::complex<float>(r, im);
        }
    }

    // instantiate the CrossMultiply class
    isce3::signal::CrossMultiply crossmulObj(length, width, upsample);

    EXPECT_EQ(crossmulObj.nrows(), length);
    EXPECT_EQ(crossmulObj.ncols(), width);
    EXPECT_EQ(crossmulObj.upsample(), upsample);
    EXPECT_GE(crossmulObj.fftsize(), width);

    // form the interferogram
    crossmulObj.crossmultiply(ifgram, ref_slc, sec_slc);

    // compute the phase difference between the computed and the expected
    // interferograms
    isce3::core::EArray2D<std::complex<float>> diff =
            ifgram * expected_ifgram.conjugate();

    // maximum phase difference
    float max_diff = diff.arg().abs().maxCoeff();

    float errtol = 1e-6;
    EXPECT_LT(max_diff, errtol);
}

TEST(CrossMultiply, FlattenInterferogram)
{

    int length = 100;
    int width = 50;
    int upsample = 1;

    isce3::core::EArray2D<std::complex<float>> ref_slc(length, width);
    isce3::core::EArray2D<std::complex<float>> sec_slc(length, width);
    isce3::core::EArray2D<double> range_offset(length, width);
    isce3::core::EArray2D<std::complex<float>> ifgram(length, width);
    isce3::core::EArray2D<std::complex<float>> expected_ifgram(length, width);
    isce3::core::EArray2D<std::complex<float>> geometry(length, width);

    double wvl = 0.23;          // meters
    double range_spacing = 7.0; // meters
    double ref_starting_range = 800000.0;
    double sec_starting_range = 800020.0;

    for (int i = 0; i < length; i++) {
        for (int j = 0; j < width; j++) {

            double ref_rng = ref_starting_range + j * range_spacing;
            double sec_rng = sec_starting_range + (j + 1) * range_spacing;
            range_offset(i, j) = -1 * (sec_rng - ref_rng) / range_spacing;
            double delay = 0.1 * std::sin(M_PI * j / width);

            // The reference SLC has only contribution from geometry
            double ref_phase = 4.0 * M_PI * ref_rng / wvl;
            ref_slc(i, j) = std::polar(1., ref_phase);

            // The secondary SLC has contributions from geometry and additional
            // delay
            double sec_phase = 4.0 * M_PI * (sec_rng + delay) / wvl;
            sec_slc(i, j) = std::polar(1., sec_phase);

            // expected interferogram after flatenning should only contain delay
            double ifgram_phase = -4.0 * M_PI * (delay) / wvl;
            expected_ifgram(i, j) = std::polar(1., ifgram_phase);
        }
    }

    isce3::signal::CrossMultiply crossmulObj(length, width, upsample);

    EXPECT_EQ(crossmulObj.nrows(), length);
    EXPECT_EQ(crossmulObj.ncols(), width);
    EXPECT_EQ(crossmulObj.upsample(), upsample);
    EXPECT_GE(crossmulObj.fftsize(), width);

    // form the interferogram and flatten it
    crossmulObj.crossmultiply(ifgram, ref_slc, sec_slc);

    // flatten the interferogram
    isce3::signal::flatten(ifgram, range_offset, range_spacing, wvl);

    // compute the phase difference between the computed and the expected
    // interferograms
    isce3::core::EArray2D<std::complex<float>> diff =
            ifgram * expected_ifgram.conjugate();

    // maximum phase difference
    float max_diff = diff.arg().abs().maxCoeff();

    float errtol = 1e-6;
    EXPECT_LT(max_diff, errtol);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
