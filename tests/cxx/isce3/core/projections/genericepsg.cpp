#include <gtest/gtest.h>
#include <isce3/core/Projections.h>
#include "projtest.h"

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

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}