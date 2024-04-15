#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <isce3/error/ErrorCode.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/StateVector.h>
#include <isce3/core/TimeDelta.h>
#include <isce3/core/Vector.h>
#include <isce3/except/Error.h>

using isce3::core::DateTime;
using isce3::core::Orbit;
using isce3::core::OrbitInterpBorderMode;
using isce3::core::OrbitInterpMethod;
using isce3::core::StateVector;
using isce3::core::TimeDelta;
using isce3::core::Vec3;

namespace isce3 { namespace core {

/** Serialize DateTime to ostream */
std::ostream & operator<<(std::ostream & os, const DateTime & dt)
{
    return os << dt.isoformat();
}

/** Serialize Vector to ostream */
std::ostream & operator<<(std::ostream & os, const Vec3 & v)
{
    return os << "{ " << v[0] << ", " << v[1] << ", " << v[2] << " }";
}

}}

/** Check if two DateTimes are equivalent to within errtol seconds */
bool compareDatetimes(const DateTime & lhs, const DateTime & rhs, double errtol)
{
    return lhs.isClose(rhs, TimeDelta(errtol));
}

/** Check if two Vectors are pointwise equivalent to within errtol */
bool compareVecs(const Vec3 & lhs, const Vec3 & rhs, double errtol)
{
    return std::abs(lhs[0] - rhs[0]) < errtol &&
           std::abs(lhs[1] - rhs[1]) < errtol &&
           std::abs(lhs[2] - rhs[2]) < errtol;
}

/** Analytical linear orbit with constant velocity */
class LinearOrbit {
public:

    LinearOrbit() = default;

    LinearOrbit(const Vec3 & initial_position, const Vec3 & velocity) :
        _initial_position(initial_position), _velocity(velocity)
    {}

    /** Get position at time t */
    Vec3 position(double t) const { return _initial_position + _velocity * t; }

    /** Get velocity at time t */
    Vec3 velocity(double /*t*/) const { return _velocity; }

private:
    Vec3 _initial_position;
    Vec3 _velocity;
};

/** Analytical orbit defined by a polynomial */
class PolynomialOrbit {
public:

    PolynomialOrbit() = default;

    PolynomialOrbit(const std::vector<Vec3> & coeffs) :
        _coeffs(coeffs), _order(int(coeffs.size()))
    {}

    /** Get position at time t */
    Vec3 position(double t) const
    {
        Vec3 position(0., 0., 0.);
        double tt = 1.;
        for (int i = 0; i < _order; ++i) {
            position += tt * _coeffs[i];
            tt *= t;
        }
        return position;
    }

    /** Get velocity at time t */
    Vec3 velocity(double t) const
    {
        Vec3 velocity(0., 0., 0.);
        double tt = 1.;
        for (int i = 1; i < _order; ++i) {
            velocity += i * tt * _coeffs[i];
            tt *= t;
        }
        return velocity;
    }

private:
    std::vector<Vec3> _coeffs;
    int _order;
};

/** Analytical circular orbit with constant angular velocity */
class CircularOrbit {
public:

    CircularOrbit() = default;

    CircularOrbit(double theta0, double phi0, double dtheta, double dphi, double r) :
        _theta0(theta0), _phi0(phi0), _dtheta(dtheta), _dphi(dphi), _r(r)
    {}

    /** Get position at time t */
    Vec3 position(double t) const
    {
        double theta = _theta0 + t * _dtheta;
        double phi = _phi0 + t * _dphi;

        double x = _r * std::cos(theta);
        double y = _r * (std::sin(theta) + std::cos(phi));
        double z = _r * std::sin(phi);

        return {x, y, z};
    }

    /** Get velocity at time t */
    Vec3 velocity(double t) const
    {
        double theta = _theta0 + t * _dtheta;
        double phi = _phi0 + t * _dphi;

        double vx = -1. * _r * _dtheta * std::sin(theta);
        double vy = _r * ((_dtheta * std::cos(theta)) - (_dphi * std::sin(phi)));
        double vz = _r * _dphi * std::cos(phi);

        return {vx, vy, vz};
    }

private:
    double _theta0, _phi0, _dtheta, _dphi, _r;
};

struct OrbitTest : public testing::Test {

    std::vector<StateVector> statevecs;

    void SetUp() override
    {
        DateTime starttime(2000, 1, 1);
        double spacing = 10.;
        int size = 11;

        Vec3 initial_position = {0., 0., 0.};
        Vec3 velocity = {4000., -1000., 4500.};
        LinearOrbit reforbit(initial_position, velocity);

        statevecs.resize(size);
        for (int i = 0; i < size; ++i) {
            double t = i * spacing;
            statevecs[i].datetime = starttime + TimeDelta(t);
            statevecs[i].position = reforbit.position(t);
            statevecs[i].velocity = reforbit.velocity(t);
        }
    }
};

TEST_F(OrbitTest, Constructor)
{
    std::string orbit_type = "POE";
    Orbit orbit(statevecs, orbit_type);

    // reference epoch defaults to time of first state vector
    DateTime refepoch = statevecs[0].datetime;
    double dt = (statevecs[1].datetime - statevecs[0].datetime).getTotalSeconds();
    int size = statevecs.size();

    EXPECT_EQ(orbit.type(), orbit_type);
    EXPECT_EQ( orbit.referenceEpoch(), refepoch );
    EXPECT_DOUBLE_EQ( orbit.spacing(), dt );
    EXPECT_EQ( orbit.size(), size );

    for (int i = 0; i < size; ++i) {
        double t = (statevecs[i].datetime - refepoch).getTotalSeconds();
        EXPECT_DOUBLE_EQ( orbit.time(i), t );
        EXPECT_EQ( orbit.position(i), statevecs[i].position );
        EXPECT_EQ( orbit.velocity(i), statevecs[i].velocity );
    }
}

TEST_F(OrbitTest, GetStateVectors)
{
    Orbit orbit(statevecs);
    std::vector<StateVector> orbit_statevecs = orbit.getStateVectors();

    EXPECT_EQ( orbit_statevecs.size(), statevecs.size() );

    int size = statevecs.size();
    double errtol = 1e-13;
    for (int i = 0; i < size; ++i) {
        DateTime t1 = orbit_statevecs[i].datetime;
        DateTime t2 = statevecs[i].datetime;
        EXPECT_PRED3( compareDatetimes, t1, t2, errtol );
        EXPECT_EQ( orbit_statevecs[i].position, statevecs[i].position );
        EXPECT_EQ( orbit_statevecs[i].velocity, statevecs[i].velocity );
    }
}

TEST_F(OrbitTest, SetStateVectors)
{
    DateTime refepoch = statevecs[0].datetime;
    double dt = (statevecs[1].datetime - statevecs[0].datetime).getTotalSeconds();
    int size = statevecs.size();

    Orbit orbit;
    orbit.referenceEpoch(refepoch);
    orbit.setStateVectors(statevecs);

    EXPECT_EQ( orbit.referenceEpoch(), refepoch );
    EXPECT_DOUBLE_EQ( orbit.spacing(), dt );
    EXPECT_EQ( orbit.size(), size );

    for (int i = 0; i < size; ++i) {
        double t = (statevecs[i].datetime - refepoch).getTotalSeconds();
        EXPECT_DOUBLE_EQ( orbit.time(i), t );
        EXPECT_EQ( orbit.position(i), statevecs[i].position );
        EXPECT_EQ( orbit.velocity(i), statevecs[i].velocity );
    }
}

TEST_F(OrbitTest, InvalidStateVectors)
{
    Orbit orbit(statevecs);

    // two or more state vectors are required
    {
        std::vector<StateVector> new_statevecs(1);
        EXPECT_THROW( orbit.setStateVectors(new_statevecs), std::invalid_argument );
    }

    // state vectors must be uniformly sampled
    {
        std::vector<StateVector> new_statevecs(3);
        new_statevecs[0].datetime = DateTime();
        new_statevecs[1].datetime = DateTime() + TimeDelta(1.);
        new_statevecs[2].datetime = DateTime() + TimeDelta(10.);
        EXPECT_THROW( orbit.setStateVectors(new_statevecs), std::invalid_argument );
    }
}

TEST_F(OrbitTest, ReferenceEpoch)
{
    Orbit orbit(statevecs);
    DateTime new_refepoch = statevecs[1].datetime;
    orbit.referenceEpoch(new_refepoch);

    EXPECT_EQ( orbit.referenceEpoch(), new_refepoch );

    double errtol = 1e-13;
    for (int i = 0; i < orbit.size(); ++i) {
        DateTime t1 = statevecs[i].datetime;
        DateTime t2 = orbit.referenceEpoch() + TimeDelta(orbit.time(i));
        EXPECT_PRED3( compareDatetimes, t1, t2, errtol );
    }
}

TEST_F(OrbitTest, InterpMethod)
{
    Orbit orbit(statevecs);
    OrbitInterpMethod new_method = OrbitInterpMethod::Legendre;
    orbit.interpMethod(new_method);

    EXPECT_EQ( orbit.interpMethod(), new_method );
}

TEST_F(OrbitTest, StartMidEndTime)
{
    // Orbit with two state vectors separated by 1 second
    {
        std::vector<StateVector> statevecs(2);
        statevecs[0].datetime = DateTime(2000, 1, 1, 0, 0, 0);
        statevecs[1].datetime = DateTime(2000, 1, 1, 0, 0, 1);
        Orbit orbit(statevecs);

        EXPECT_DOUBLE_EQ( orbit.startTime(), 0. );
        EXPECT_DOUBLE_EQ( orbit.midTime(), 0.5 );
        EXPECT_DOUBLE_EQ( orbit.endTime(), 1. );
    }

    // Orbit with three state vectors with 1 second spacing
    {
        std::vector<StateVector> statevecs(3);
        statevecs[0].datetime = DateTime(2000, 1, 1, 0, 0, 0);
        statevecs[1].datetime = DateTime(2000, 1, 1, 0, 0, 1);
        statevecs[2].datetime = DateTime(2000, 1, 1, 0, 0, 2);
        Orbit orbit(statevecs);

        EXPECT_DOUBLE_EQ( orbit.startTime(), 0. );
        EXPECT_DOUBLE_EQ( orbit.midTime(), 1. );
        EXPECT_DOUBLE_EQ( orbit.endTime(), 2. );
    }
}

TEST_F(OrbitTest, StartMidEndDateTime)
{
    double errtol = 1e-13;

    // Orbit with two state vectors separated by 1 second
    {
        std::vector<StateVector> statevecs(2);
        statevecs[0].datetime = DateTime(2000, 1, 1, 0, 0, 0);
        statevecs[1].datetime = DateTime(2000, 1, 1, 0, 0, 1);
        Orbit orbit(statevecs);

        EXPECT_PRED3( compareDatetimes, orbit.startDateTime(), DateTime(2000, 1, 1, 0, 0, 0), errtol );
        EXPECT_PRED3( compareDatetimes, orbit.midDateTime(), DateTime(2000, 1, 1, 0, 0, 0.5), errtol );
        EXPECT_PRED3( compareDatetimes, orbit.endDateTime(), DateTime(2000, 1, 1, 0, 0, 1), errtol );
    }

    // Orbit with three state vectors with 1 second spacing
    {
        std::vector<StateVector> statevecs(3);
        statevecs[0].datetime = DateTime(2000, 1, 1, 0, 0, 0);
        statevecs[1].datetime = DateTime(2000, 1, 1, 0, 0, 1);
        statevecs[2].datetime = DateTime(2000, 1, 1, 0, 0, 2);
        Orbit orbit(statevecs);

        EXPECT_PRED3( compareDatetimes, orbit.startDateTime(), DateTime(2000, 1, 1, 0, 0, 0), errtol );
        EXPECT_PRED3( compareDatetimes, orbit.midDateTime(), DateTime(2000, 1, 1, 0, 0, 1), errtol );
        EXPECT_PRED3( compareDatetimes, orbit.endDateTime(), DateTime(2000, 1, 1, 0, 0, 2), errtol );
    }
}

TEST_F(OrbitTest, Comparison)
{
    Orbit orbit1(statevecs);
    Orbit orbit2(statevecs);
    Orbit orbit3;

    std::string orbit_type_poe = "POE";
    Orbit orbit4(statevecs, orbit_type_poe);

    EXPECT_TRUE( orbit1 == orbit2 );
    EXPECT_TRUE( orbit1 != orbit3 );
    EXPECT_TRUE( orbit1 != orbit4 );
    EXPECT_TRUE( orbit3 != orbit4 );
}

TEST_F(OrbitTest, OrbitInterpBorderMode)
{
    Orbit orbit(statevecs);

    // throw exception on attempt to interpolate outside orbit domain
    {
        OrbitInterpBorderMode border_mode = OrbitInterpBorderMode::Error;

        double t = orbit.endTime() + 1.;
        Vec3 pos, vel;
        EXPECT_THROW( orbit.interpolate(&pos, &vel, t, border_mode), isce3::except::OutOfRange );
    }

    // output NaN on attempt to interpolate outside orbit domain
    {
        OrbitInterpBorderMode border_mode = OrbitInterpBorderMode::FillNaN;

        double t = orbit.endTime() + 1.;
        Vec3 pos, vel;
        orbit.interpolate(&pos, &vel, t, border_mode);

        EXPECT_TRUE( std::isnan(pos[0]) && std::isnan(pos[1]) && std::isnan(pos[2]) );
        EXPECT_TRUE( std::isnan(vel[0]) && std::isnan(vel[1]) && std::isnan(vel[2]) );
    }
}

struct LinearOrbitInterpTest : public testing::Test {

    LinearOrbit reforbit;
    std::vector<StateVector> statevecs;
    std::vector<double> interp_times;
    double errtol;

    void SetUp() override
    {
        DateTime starttime(2000, 1, 1);
        double spacing = 10.;
        int size = 11;

        Vec3 initial_position = {0., 0., 0.};
        Vec3 velocity = {4000., -1000., 4500.};
        reforbit = LinearOrbit(initial_position, velocity);

        statevecs.resize(size);
        for (int i = 0; i < size; ++i) {
            double t = i * spacing;
            statevecs[i].datetime = starttime + TimeDelta(t);
            statevecs[i].position = reforbit.position(t);
            statevecs[i].velocity = reforbit.velocity(t);
        }

        interp_times = { 23.3, 36.7, 54.5, 89.3 };
        errtol = 1e-8;
    }
};

TEST_F(LinearOrbitInterpTest, Hermite)
{
    Orbit orbit(statevecs, OrbitInterpMethod::Hermite);

    for (auto t : interp_times) {
        Vec3 pos, vel;
        orbit.interpolate(&pos, &vel, t);
        EXPECT_PRED3( compareVecs, pos, reforbit.position(t), errtol );
        EXPECT_PRED3( compareVecs, vel, reforbit.velocity(t), errtol );
    }
}

TEST_F(LinearOrbitInterpTest, Legendre)
{
    Orbit orbit(statevecs, OrbitInterpMethod::Legendre);

    for (auto t : interp_times) {
        Vec3 pos, vel;
        orbit.interpolate(&pos, &vel, t);
        EXPECT_PRED3( compareVecs, pos, reforbit.position(t), errtol );
        EXPECT_PRED3( compareVecs, vel, reforbit.velocity(t), errtol );
    }
}

struct PolynomialOrbitInterpTest : public testing::Test {

    PolynomialOrbit reforbit;
    std::vector<StateVector> statevecs;
    std::vector<double> interp_times;
    double errtol;

    void SetUp() override
    {
        DateTime starttime(2000, 1, 1);
        double spacing = 10.;
        int size = 11;

        std::vector<Vec3> coeffs =
            {{ -7000000.0, 5400000.0 ,    0.0},
             {     5435.0,   -4257.0 , 7000.0},
             {      -45.0,      23.0 ,   11.0},
             {        7.3,       3.9 ,    0.0},
             {        0.0,       0.01,    0.0}};
        reforbit = PolynomialOrbit(coeffs);

        statevecs.resize(size);
        for (int i = 0; i < size; ++i) {
            double t = i * spacing;
            statevecs[i].datetime = starttime + TimeDelta(t);
            statevecs[i].position = reforbit.position(t);
            statevecs[i].velocity = reforbit.velocity(t);
        }

        interp_times = { 23.3, 36.7, 54.5, 89.3 };
        errtol = 1e-8;
    }
};

TEST_F(PolynomialOrbitInterpTest, Hermite)
{
    Orbit orbit(statevecs, OrbitInterpMethod::Hermite);

    for (auto t : interp_times) {
        Vec3 pos, vel;
        orbit.interpolate(&pos, &vel, t);
        EXPECT_PRED3( compareVecs, pos, reforbit.position(t), errtol );
        EXPECT_PRED3( compareVecs, vel, reforbit.velocity(t), errtol );
    }
}

TEST_F(PolynomialOrbitInterpTest, Legendre)
{
    Orbit orbit(statevecs, OrbitInterpMethod::Legendre);

    for (auto t : interp_times) {
        Vec3 pos, vel;
        orbit.interpolate(&pos, &vel, t);
        EXPECT_PRED3( compareVecs, pos, reforbit.position(t), errtol );
        EXPECT_PRED3( compareVecs, vel, reforbit.velocity(t), errtol );
    }
}

struct CircularOrbitInterpTest : public testing::Test {

    CircularOrbit reforbit;
    std::vector<StateVector> statevecs;
    std::vector<double> interp_times;
    double errtol;

    void SetUp() override
    {
        DateTime starttime(2000, 1, 1);
        double spacing = 5.;
        int size = 11;

        double theta0 = 2. * M_PI / 8.;
        double phi0 = 2. * M_PI / 12.;
        double dtheta = 2. * M_PI / 7000.;
        double dphi = 2. * M_PI / 4000.;
        double r = 8000000.;
        reforbit = CircularOrbit(theta0, phi0, dtheta, dphi, r);

        statevecs.resize(size);
        for (int i = 0; i < size; ++i) {
            double t = i * spacing;
            statevecs[i].datetime = starttime + TimeDelta(t);
            statevecs[i].position = reforbit.position(t);
            statevecs[i].velocity = reforbit.velocity(t);
        }

        interp_times = { 11.65, 18.35, 27.25, 44.65 };
        errtol = 1e-8;
    }
};

TEST_F(CircularOrbitInterpTest, Hermite)
{
    Orbit orbit(statevecs, OrbitInterpMethod::Hermite);

    for (auto t : interp_times) {
        Vec3 pos, vel;
        orbit.interpolate(&pos, &vel, t);
        EXPECT_PRED3( compareVecs, pos, reforbit.position(t), errtol );
        EXPECT_PRED3( compareVecs, vel, reforbit.velocity(t), errtol );
    }
}

TEST_F(CircularOrbitInterpTest, Legendre)
{
    Orbit orbit(statevecs, OrbitInterpMethod::Legendre);

    for (auto t : interp_times) {
        Vec3 pos, vel;
        orbit.interpolate(&pos, &vel, t);
        EXPECT_PRED3( compareVecs, pos, reforbit.position(t), errtol );
        EXPECT_PRED3( compareVecs, vel, reforbit.velocity(t), errtol );
    }
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
