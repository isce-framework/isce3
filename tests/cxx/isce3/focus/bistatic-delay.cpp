#include <gtest/gtest.h>

#include <isce3/core/Constants.h>
#include <isce3/core/Vector.h>
#include <isce3/focus/BistaticDelay.h>

using isce3::core::Vec3;
using isce3::focus::bistaticDelay;

/** Analytical linear orbit with constant velocity */
class LinearOrbit {
public:
    LinearOrbit(const Vec3 & initial_position, const Vec3 & velocity)
        : _initial_position(initial_position), _velocity(velocity) {}

    /** Get position at time t */
    Vec3 position(double t) const { return _initial_position + _velocity * t; }

    /** Get velocity at time t */
    Vec3 velocity(double /*t*/) const { return _velocity; }

private:
    Vec3 _initial_position;
    Vec3 _velocity;
};

TEST(BistaticDelay, BistaticDelay)
{
    // platform orbit
    Vec3 initial_position = {0., 0., 700'000.};
    Vec3 velocity = {0., 8000., 0.};
    LinearOrbit orbit(initial_position, velocity);

    // target position
    Vec3 x = {50'000., 20'000., 0.};

    for (int i = 0; i < 11; ++i) {
        auto t = static_cast<double>(i);

        // compute bistatic delay term (tau)
        Vec3 p = orbit.position(t);
        Vec3 v = orbit.velocity(t);
        double tau = bistaticDelay(p, v, x);

        // compare to roundtrip delay using platform position at time = t + tau
        Vec3 p1 = orbit.position(t + tau);
        double d = (x - p).norm() + (p1 - x).norm();
        double dt = d / isce3::core::speed_of_light;

        EXPECT_DOUBLE_EQ(tau, dt);
    }
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
