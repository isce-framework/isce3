#include <isce/orbit_wip/Orbit.h>
#include <gtest/gtest.h>

// XXX ambiguous: core::Orbit vs. orbit_wip::Orbit
// using namespace isce::core;
using isce::core::DateTime;
using isce::core::StateVector;
using isce::core::TimeDelta;

using isce::orbit_wip::Orbit;

bool isclose(const Vec3 & lhs, const Vec3 & rhs)
{
    double errtol = 1e-6;
    return std::abs(lhs[0] - rhs[0]) < errtol &&
           std::abs(lhs[1] - rhs[1]) < errtol &&
           std::abs(lhs[2] - rhs[2]) < errtol;
}

namespace isce { namespace core {
std::ostream & operator<<(std::ostream & os, const isce::core::Vec3 & v)
{
    return os << std::endl << "{ " << v[0] << ", " << v[1] << ", " << v[2] << " }";
}
} }

// get state vector from linear orbit
StateVector make_linear_statevec(
        const DateTime & refepoch,
        const TimeDelta & dt,
        const Vec3 & initial_pos,
        const Vec3 & velocity)
{
    double _dt = dt.getTotalSeconds();
    return {refepoch + dt, initial_pos + _dt * velocity, velocity};
}

// make orbit with linear platform trajectory
Orbit make_linear_orbit(
        const DateTime & refepoch,
        const TimeDelta & spacing,
        const Vec3 & initial_pos,
        const Vec3 & velocity,
        int size)
{
    Orbit orbit (refepoch, spacing, size);

    for (int i = 0; i < size; ++i) {
        TimeDelta dt = spacing * i;
        orbit[i] = make_linear_statevec(refepoch, dt, initial_pos, velocity);
    }

    return orbit;
}

struct InterpOrbitTest_Linear : public testing::Test {
    Orbit orbit;
    std::vector<double> test_times;
    std::vector<StateVector> expected;

    void SetUp() override
    {
        // 11 state vectors spaced 10s apart
        DateTime refepoch (2000, 1, 1);
        TimeDelta spacing = 10.;
        Vec3 initial_pos {0., 0., 0.};
        Vec3 velocity {4000., -1000., 4500.};
        int size = 11;

        orbit = make_linear_orbit(refepoch, spacing, initial_pos, velocity, size);

        test_times = {23.3, 36.7, 54.5, 89.3};

        for (size_t i = 0; i < test_times.size(); ++i) {
            TimeDelta dt = test_times[i];
            StateVector sv = make_linear_statevec(refepoch, dt, initial_pos, velocity);
            expected.push_back(sv);
        }
    }
};

TEST_F(InterpOrbitTest_Linear, HermiteInterpolate)
{
    for (size_t i = 0; i < test_times.size(); ++i) {
        Vec3 position, velocity;
        hermite_interpolate(orbit, test_times[i], &position, &velocity);

        EXPECT_PRED2( isclose, position, expected[i].position );
        EXPECT_PRED2( isclose, velocity, expected[i].velocity );
    }
}

TEST_F(InterpOrbitTest_Linear, LegendreInterpolate)
{
    for (size_t i = 0; i < test_times.size(); ++i) {
        Vec3 position, velocity;
        legendre_interpolate(orbit, test_times[i], &position, &velocity);

        EXPECT_PRED2( isclose, position, expected[i].position );
        EXPECT_PRED2( isclose, velocity, expected[i].velocity );
    }
}

TEST_F(InterpOrbitTest_Linear, SCHInterpolate)
{
    for (size_t i = 0; i < test_times.size(); ++i) {
        Vec3 position, velocity;
        sch_interpolate(orbit, test_times[i], &position, &velocity);

        EXPECT_PRED2( isclose, position, expected[i].position );
        EXPECT_PRED2( isclose, velocity, expected[i].velocity );
    }
}

// get state vector from orbit defined by polynomial
StateVector make_polynomial_statevec(
        const DateTime & refepoch,
        const TimeDelta & dt,
        const std::vector<Vec3> & coeffs)
{
    double _dt = dt.getTotalSeconds();

    int order = coeffs.size();

    Vec3 pos = {0., 0., 0.};
    double k = 1.;
    for (int i = 0; i < order; ++i) {
        pos += k * coeffs[i];
        k *= _dt;
    }

    Vec3 vel = {0., 0., 0.};
    k = 1.;
    for (int i = 1; i < order; ++i) {
        vel += i * k * coeffs[i];
        k *= _dt;
    }

    return {refepoch + dt, pos, vel};
}

// make orbit with platform trajectory defined by polynomial
Orbit make_polynomial_orbit(
        const DateTime & refepoch,
        const TimeDelta & spacing,
        const std::vector<Vec3> & coeffs,
        int size)
{
    Orbit orbit (refepoch, spacing, size);

    for (int i = 0; i < size; ++i) {
        TimeDelta dt = spacing * i;
        orbit[i] = make_polynomial_statevec(refepoch, dt, coeffs);
    }

    return orbit;
}

struct InterpOrbitTest_Polynomial : public testing::Test {
    Orbit orbit;
    std::vector<double> test_times;
    std::vector<StateVector> expected;

    void SetUp() override
    {
        // 11 state vectors spaced 10s apart
        DateTime refepoch (2000, 1, 1);
        TimeDelta spacing = 10.;
        std::vector<Vec3> coeffs = {
            {-7000000., 5400000., 0.},
            {5435., -4257., 7000.},
            {-45., 23., 11.},
            {7.3, 3.9, 0.},
            {0., 0.01, 0.} };
        int size = 11;

        orbit = make_polynomial_orbit(refepoch, spacing, coeffs, size);

        test_times = {23.3, 36.7, 54.5, 89.3};

        for (size_t i = 0; i < test_times.size(); ++i) {
            TimeDelta dt = test_times[i];
            StateVector sv = make_polynomial_statevec(refepoch, dt, coeffs);
            expected.push_back(sv);
        }
    }
};

TEST_F(InterpOrbitTest_Polynomial, HermiteInterpolate)
{
    for (size_t i = 0; i < test_times.size(); ++i) {
        Vec3 position, velocity;
        hermite_interpolate(orbit, test_times[i], &position, &velocity);

        EXPECT_PRED2( isclose, position, expected[i].position );
        EXPECT_PRED2( isclose, velocity, expected[i].velocity );
    }
}

TEST_F(InterpOrbitTest_Polynomial, LegendreInterpolate)
{
    for (size_t i = 0; i < test_times.size(); ++i) {
        Vec3 position, velocity;
        legendre_interpolate(orbit, test_times[i], &position, &velocity);

        EXPECT_PRED2( isclose, position, expected[i].position );
        EXPECT_PRED2( isclose, velocity, expected[i].velocity );
    }
}

TEST_F(InterpOrbitTest_Polynomial, SCHInterpolate)
{
    for (size_t i = 0; i < test_times.size(); ++i) {
        Vec3 position, velocity;
        sch_interpolate(orbit, test_times[i], &position, &velocity);

        EXPECT_PRED2( isclose, position, expected[i].position );
        EXPECT_PRED2( isclose, velocity, expected[i].velocity );
    }
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

