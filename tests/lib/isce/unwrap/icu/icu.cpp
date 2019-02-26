#include <cmath> // cos, sin, sqrt, fmod
#include <complex> // std::complex, std::arg
#include <cstdint> // uint8_t
#include <gtest/gtest.h> // TEST, ASSERT_EQ, ASSERT_TRUE, testing::InitGoogleTest, RUN_ALL_TESTS
#include <valarray> // std::valarray, std::abs

#include "isce/unwrap/icu/ICU.h" // isce::unwrap::icu::ICU
#include "isce/io/Raster.h" // isce::io::Raster

TEST(ICU, GetSetters)
{
    isce::unwrap::icu::ICU icuobj;

    icuobj.numBufLines(1024);
    ASSERT_EQ(icuobj.numBufLines(), 1024);
    icuobj.numOverlapLines(50);
    ASSERT_EQ(icuobj.numOverlapLines(), 50);
    icuobj.usePhaseGradNeut(true);
    ASSERT_EQ(icuobj.usePhaseGradNeut(), true);
    icuobj.useIntensityNeut(true);
    ASSERT_EQ(icuobj.useIntensityNeut(), true);
    icuobj.phaseGradWinSize(3);
    ASSERT_EQ(icuobj.phaseGradWinSize(), 3);
    icuobj.neutPhaseGradThr(1.5f);
    ASSERT_EQ(icuobj.neutPhaseGradThr(), 1.5f);
    icuobj.neutIntensityThr(4.f);
    ASSERT_EQ(icuobj.neutIntensityThr(), 4.f);
    icuobj.neutCorrThr(0.5f);
    ASSERT_EQ(icuobj.neutCorrThr(), 0.5f);
    icuobj.numTrees(3);
    ASSERT_EQ(icuobj.numTrees(), 3);
    icuobj.maxBranchLen(32);
    ASSERT_EQ(icuobj.maxBranchLen(), 32);
    icuobj.ratioDxDy(2.f);
    ASSERT_EQ(icuobj.ratioDxDy(), 2.f);
    icuobj.initCorrThr(0.4f);
    ASSERT_EQ(icuobj.initCorrThr(), 0.4f);
    icuobj.maxCorrThr(0.8f);
    ASSERT_EQ(icuobj.maxCorrThr(), 0.8f);
    icuobj.corrThrInc(0.2f);
    ASSERT_EQ(icuobj.corrThrInc(), 0.2f);
    icuobj.minCCAreaFrac(0.01f);
    ASSERT_EQ(icuobj.minCCAreaFrac(), 0.01f);
    icuobj.numBsLines(8);
    ASSERT_EQ(icuobj.numBsLines(), 8);
    icuobj.minBsPts(12);
    ASSERT_EQ(icuobj.minBsPts(), 12);
    icuobj.bsPhaseVarThr(3.f);
    ASSERT_EQ(icuobj.bsPhaseVarThr(), 3.f);
}

TEST(ICU, ResidueCalculation)
{
    // (Example taken from Appendix B of Sean Buckley's Dissertation)
    std::valarray<float> phase({ 0.0f, 0.3f, 0.4f, 0.1f, 
                                 0.8f, 0.6f, 0.7f, 0.7f });
    phase *= 2.f * M_PI;

    constexpr size_t l = 2;
    constexpr size_t w = 4;
    std::valarray<signed char> charge(l*w), refcharge(l*w);
    refcharge[0] = 1;
    refcharge[2] = -1;

    isce::unwrap::icu::ICU icuobj;
    icuobj.getResidues(&charge[0], &phase[0], l, w);
    ASSERT_TRUE((charge == refcharge).min());
}

TEST(ICU, PhaseGradNeutronCalculation)
{
    isce::unwrap::icu::ICU icuobj;
    icuobj.usePhaseGradNeut(true);
    icuobj.phaseGradWinSize(3);

    constexpr size_t l = 4;
    constexpr size_t w = 4;
    std::valarray<std::complex<float>> intf(l*w);
    std::valarray<bool> neut(l*w), refneut(l*w);

    // Interferogram with phase slope = 2.9 rad/sample (below threshold)
    float dphi = 2.9f;
    for (size_t i = 0; i < w; ++i)
    {
        std::complex<float> z{cos(dphi * float(i)), sin(dphi * float(i))};
        for (size_t j = 0; j < l; ++j) { intf[j * w + i] = z; }
    }

    icuobj.genNeutrons(&neut[0], &intf[0], nullptr, l, w);
    ASSERT_TRUE((neut == refneut).min());

    // Interferogram with phase slope = 3.1 rad/sample (above threshold)
    dphi = 3.1f;
    for (size_t i = 0; i < w; ++i)
    {
        std::complex<float> z{cos(dphi * float(i)), sin(dphi * float(i))};
        for (size_t j = 0; j < l; ++j) { intf[j * w + i] = z; }
    }
    refneut[2 * w + 2] = true;

    icuobj.genNeutrons(&neut[0], &intf[0], nullptr, l, w);
    ASSERT_TRUE((neut == refneut).min());
}

TEST(ICU, IntensityNeutronCalculation)
{
    isce::unwrap::icu::ICU icuobj;
    icuobj.useIntensityNeut(true);

    constexpr size_t l = 64;
    constexpr size_t w = 128;
    std::valarray<std::complex<float>> intf(l*w);
    std::valarray<float> corr(0.9f, l*w);
    std::valarray<bool> neut(l*w), refneut(l*w);

    // High intensity, high correlation (not neutron)
    std::complex<float> z{sqrt(5.f), sqrt(5.f)};
    intf[0] = z;

    // Low intensity, low correlation (not neutron)
    corr[1] = 0.2f;

    // High intensity, low correlation (neutron)
    intf[2] = z;
    corr[2] = 0.2f;
    refneut[2] = true;

    icuobj.genNeutrons(&neut[0], &intf[0], &corr[0], l, w);
    ASSERT_TRUE((neut == refneut).min());
}

TEST(ICU, TreeGrowing)
{
    isce::unwrap::icu::ICU icuobj;
    icuobj.numTrees(1);
    icuobj.maxBranchLen(16);

    constexpr size_t l = 100;
    constexpr size_t w = 100;
    std::valarray<signed char> charge(l*w);
    std::valarray<bool> neut(l*w), tree(l*w), reftree(l*w);

    // Load reference raster.
    isce::io::Raster refRaster("../../data/icu/tree");
    ASSERT_TRUE(refRaster.length() == l && refRaster.width() == w);
    refRaster.getBlock(reinterpret_cast<uint8_t *>(&reftree[0]), 0, 0, w, l);

    // One residue near each of the four edges that each discharge via a branch 
    // cut to the nearest edge
    charge[ 5 * w + 50] = 1;
    charge[95 * w + 50] = -1;
    charge[50 * w +  5] = -1;
    charge[50 * w + 95] = 1;

    // Two residues connected by a single branch cut which neutralize each other 
    // (twig should discharge before reaching neutron)
    charge[40 * w + 35] = 1;
    charge[50 * w + 35] = -1;
    neut[62 * w + 35] = true;
    
    // Two residues connected via neutrons (twig reaches max search radius 
    // without discharging)
    charge[65 * w + 75] = 1;
    neut[65 * w + 65] = true;
    neut[60 * w + 65] = true;
    neut[50 * w + 65] = true;
    neut[50 * w + 75] = true;
    neut[45 * w + 75] = true;
    neut[35 * w + 75] = true;
    charge[35 * w + 65] = 1;

    icuobj.growTrees(&tree[0], &charge[0], &neut[0], l, w);
    ASSERT_TRUE((tree == reftree).min());
}

TEST(ICU, RunICU)
{
    constexpr size_t l = 1100;
    constexpr size_t w = 256;

    // Interferogram with simple vertical phase gradient
    std::valarray<std::complex<float>> intf(l*w);
    for (size_t j = 0; j < l; ++j)
    {
        float y = float(j) / float(l) * 50.f;
        std::complex<float> z{cos(y), sin(y)};
        for (size_t i = 0; i < w; ++i) { intf[j * w + i] = z; }
    }

    // Correlation (results in one tall connected component with "U" shape and 
    // a separate rectangular component)
    std::valarray<float> corr(l*w);
    for (size_t j = 100; j < 900; ++j)
    {
        for (size_t i = 50; i < 100; ++i) { corr[j * w + i] = 1.f; }
        for (size_t i = 150; i < 200; ++i) { corr[j * w + i] = 1.f; }
    }
    for (size_t j = 900; j < 950; ++j)
    {
        for (size_t i = 50; i < 200; ++i) { corr[j * w + i] = 1.f; }
    }

    for (size_t j = 1000; j < 1050; ++j)
    {
        for (size_t i = 50; i < 200; ++i) { corr[j * w + i] = 1.f; }
    }

    // Create interferogram, correlation rasters
    isce::io::Raster intfRaster("./intf", w, l, 1, GDT_CFloat32, "ENVI");
    intfRaster.setBlock(intf, 0, 0, w, l);
    isce::io::Raster corrRaster("./corr", w, l, 1, GDT_Float32, "ENVI");
    corrRaster.setBlock(corr, 0, 0, w, l);

    // Init output unwrapped phase, connected component labels rasters
    isce::io::Raster unwRaster("./unw", w, l, 1, GDT_Float32, "ENVI");
    isce::io::Raster cclRaster("./ccl", w, l, 1, GDT_Byte, "ENVI");

    // Configure ICU to process the interferogram as 3 tiles.
    isce::unwrap::icu::ICU icuobj;
    icuobj.numBufLines(400);
    icuobj.numOverlapLines(50);

    icuobj.unwrap(unwRaster, cclRaster, intfRaster, corrRaster);
}

TEST(ICU, CheckUnwrappedPhase)
{
    // Read interferogram from prior test.
    isce::io::Raster intfRaster("./intf");
    const size_t l = intfRaster.length();
    const size_t w = intfRaster.width();
    std::valarray<std::complex<float>> intf(l*w);
    intfRaster.getBlock(intf, 0, 0, w, l);

    // Get wrapped phase.
    std::valarray<float> phase(l*w);
    for (size_t i = 0; i < l*w; ++i) { phase[i] = std::arg(intf[i]); }

    // Read unwrapped phase from prior test.
    isce::io::Raster unwRaster("./unw");
    ASSERT_TRUE(unwRaster.length() == l && unwRaster.width() == w);
    std::valarray<float> unw(l*w);
    unwRaster.getBlock(unw, 0, 0, w, l);

    // Read connected component labels from prior test.
    isce::io::Raster cclRaster("./ccl");
    ASSERT_TRUE(cclRaster.length() == l && cclRaster.width() == w);
    std::valarray<uint8_t> ccl(l*w);
    cclRaster.getBlock(ccl, 0, 0, w, l);

    // Check sine distance between unwrapped and wrapped phase within connected 
    // component(s).
    std::valarray<float> sindist(l*w);
    for (size_t i = 0; i < l*w; ++i) 
    { 
        if (ccl[i] != 0) { sindist[i] = std::abs(sin(unw[i] - phase[i])); }
    }
    ASSERT_TRUE(sindist.max() < 1e-5);
}

TEST(ICU, CheckConnCompLabels)
{
    constexpr size_t l = 1100;
    constexpr size_t w = 256;

    // Get reference labels.
    std::valarray<uint8_t> refccl(l*w);
    for (size_t j = 100; j < 900; ++j)
    {
        for (size_t i = 50; i < 100; ++i) { refccl[j * w + i] = 1; }
        for (size_t i = 150; i < 200; ++i) { refccl[j * w + i] = 1; }
    }
    for (size_t j = 900; j < 950; ++j)
    {
        for (size_t i = 50; i < 200; ++i) { refccl[j * w + i] = 1; }
    }

    // (Note: connected component with label "2" gets merged with component "1" 
    // during the labelling process.)
    for (size_t j = 1000; j < 1050; ++j)
    {
        for (size_t i = 50; i < 200; ++i) { refccl[j * w + i] = 3; }
    }

    // Read connected component labels from prior test.
    isce::io::Raster cclRaster("./ccl");
    ASSERT_TRUE(cclRaster.length() == l && cclRaster.width() == w);
    std::valarray<uint8_t> ccl(l*w);
    cclRaster.getBlock(ccl, 0, 0, w, l);

    ASSERT_TRUE((ccl == refccl).min());
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

