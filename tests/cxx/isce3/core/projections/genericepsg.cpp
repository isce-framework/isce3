#include <gtest/gtest.h>
#include <isce3/core/Projections.h>
#include "projtest.h"

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

isce3::core::GenericEPSG Can(3979);
isce3::core::GenericEPSG Aus(3577);
isce3::core::GenericEPSG Kor(5186);

struct GenericEPSGTest : public ::testing::Test {
    unsigned int fails;
    virtual void SetUp() { fails = 0; }
    virtual void TearDown() {
        if (fails > 0) {
            std::cerr << "GenericEPSG::TearDown sees failures" << std::endl;
        }
    }
};

#define epsgTest(...) PROJ_TEST(GenericEPSGTest, __VA_ARGS__)
#define makeprojTest(...) MAKE_PROJ_TEST(GenericEPSGTest, __VA_ARGS__)

// This is a NAD-83 system so there is a small offset from WGS84
epsgTest(Can, CanOrigin,
        {-1.6580629422844335, 0.855211480709109, 0.},
        {0., 0., 0.});
epsgTest(Aus, AusOrigin,
        {132.0*M_PI/180.0, 0., 0.},
        {0.,0.,0.});
epsgTest(Kor, KorOrigin,
        {127.0*M_PI/180.0, 38.0*M_PI/180.0, 0.},
        {200000.0, 600000.0, 0.0});

epsgTest(Can, Can1,
        {-1.553988006969734, 0.9153985328506767, 0.},
        {4.0e+05, 4.0e+05, 0.});

epsgTest(Can, Can2,
        {-1.7475929794365734, 0.884407195599607, 0.},
        {-3.6e+05, 2.0e+05, 0.});

epsgTest(Can, Can3,
        {-1.3766583225123716, 0.7120684078059226, 0.},
        {1.4e+06, -7.5e+05, 0.});

//Australia tests
epsgTest(Aus, Aus1,
        {2.326601641674536, -0.59163508189832, 0.},
        {1.2e+05, -3.7e+06, 0.});

epsgTest(Aus, Aus2,
        {2.313562879599467, -0.27900251672315923, 0.},
        {6.0e+04, -1.7e+06, 0.});

epsgTest(Aus, Aus3,
        {2.344362841755795, -0.7993597057678191, 0.},
        {1.9e+05, -5e+06, 0.});

//South Korea tests - This only appears to be good to 1cm
epsgTest(Kor, Kor1,
        {2.2412392328140913, 0.6001667563213443, 0.},
        {3.3e+05, 2.0e+05, 0.});

epsgTest(Kor, Kor2,
        {2.135176403804947, 0.6930318564897303, 0.},
        {-2.0e+05, 8.0e+05, 0.});

epsgTest(Kor, Kor3,
        {2.103517202529827, 0.6836580237834484, 0.},
        {-3.6e+05, 7.5e+05, 0.});

// Test creating projections
makeprojTest(3979, makeCanada);
makeprojTest(3577, makeAustralia);
makeprojTest(5176, makeSouthKorea);

// Concurrent forward/inverse calls on a shared object from OpenMP threads
// must match serial results.
static int roundTripErrors(const isce3::core::GenericEPSG& proj,
                           const std::vector<isce3::core::Vec3>& llh,
                           const std::vector<isce3::core::Vec3>& ref, int i)
{
    isce3::core::Vec3 xyz, back;
    if (proj.forward(llh[i], xyz) != 0 || proj.inverse(xyz, back) != 0 ||
        (xyz - ref[i]).norm() > 1e-6 ||
        std::abs(back[0] - llh[i][0]) > 1e-10 ||
        std::abs(back[1] - llh[i][1]) > 1e-10 ||
        std::abs(back[2] - llh[i][2]) > 1e-6) {
        return 1;
    }
    return 0;
}

struct GenericEPSGThreading : public ::testing::Test {
    const isce3::core::GenericEPSG proj {3979};
    const int npts = 20000;
    std::vector<isce3::core::Vec3> llh, ref;

    void SetUp() override
    {
        llh.resize(npts);
        ref.resize(npts);
        for (int i = 0; i < npts; ++i) {
            llh[i] = {(-120.0 + 50.0 * i / npts) * M_PI / 180.0,
                      (45.0 + 15.0 * i / npts) * M_PI / 180.0,
                      10.0 * i / npts};
            ASSERT_EQ(proj.forward(llh[i], ref[i]), 0);
        }
    }
};

TEST_F(GenericEPSGThreading, ParallelFor)
{
    int errors = 0;
    #pragma omp parallel for reduction(+:errors)
    for (int i = 0; i < npts; ++i) {
        errors += roundTripErrors(proj, llh, ref, i);
    }
    EXPECT_EQ(errors, 0);
}

TEST_F(GenericEPSGThreading, MoreThreadsThanMax)
{
    int errors = 0;
    #ifdef _OPENMP
    const int nthreads = 2 * omp_get_max_threads() + 1;
    #else
    const int nthreads = 1;
    #endif
    #pragma omp parallel for num_threads(nthreads) reduction(+:errors)
    for (int i = 0; i < npts; ++i) {
        errors += roundTripErrors(proj, llh, ref, i);
    }
    EXPECT_EQ(errors, 0);
}

TEST_F(GenericEPSGThreading, NestedParallel)
{
    int errors = 0;
    #ifdef _OPENMP
    const int prev_levels = omp_get_max_active_levels();
    omp_set_max_active_levels(2);
    #endif
    #pragma omp parallel for num_threads(4) reduction(+:errors)
    for (int j = 0; j < 4; ++j) {
        #pragma omp parallel for num_threads(4) reduction(+:errors)
        for (int i = j; i < npts; i += 4) {
            errors += roundTripErrors(proj, llh, ref, i);
        }
    }
    #ifdef _OPENMP
    omp_set_max_active_levels(prev_levels);
    #endif
    EXPECT_EQ(errors, 0);
}

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}