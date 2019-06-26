#include <isce/cuda/orbit_wip/Orbit.h>
#include <isce/orbit_wip/Orbit.h>
#include <isce/except/Error.h>
#include <gtest/gtest.h>

struct OrbitTest : public testing::Test {

    std::vector<isce::core::StateVector> statevecs;

    isce::core::DateTime refepoch;
    isce::core::TimeDelta spacing;
    int size;

    void SetUp() override
    {
        statevecs.resize(2);

        statevecs[0].datetime = "2016-04-08T09:13:13.000000";
        statevecs[0].position = {-3752316.976337, 4925051.878499, 3417259.473609};
        statevecs[0].velocity = {3505.330104, -1842.136554, 6482.122476};

        statevecs[1].datetime = "2016-04-08T09:13:23.000000";
        statevecs[1].position = {-3717067.52658, 4906329.056304, 3481886.455117};
        statevecs[1].velocity = {3544.479224, -1902.402281, 6443.152265};

        refepoch = statevecs[0].datetime;
        spacing = statevecs[1].datetime - statevecs[0].datetime;
        size = statevecs.size();
    }
};

TEST_F(OrbitTest, FromStateVectors)
{
    typedef isce::cuda::orbit_wip::Orbit Orbit;

    Orbit orbit = Orbit::from_statevectors(statevecs);

    {
        isce::core::Vec3 pos = orbit.position()[0];
        EXPECT_EQ( pos, statevecs[0].position );
        isce::core::Vec3 vel = orbit.velocity()[0];
        EXPECT_EQ( vel, statevecs[0].velocity );
    }

    {
        isce::core::Vec3 pos = orbit.position()[1];
        EXPECT_EQ( pos, statevecs[1].position );
        isce::core::Vec3 vel = orbit.velocity()[1];
        EXPECT_EQ( vel, statevecs[1].velocity );
    }
}

TEST_F(OrbitTest, BasicConstructor)
{
    isce::cuda::orbit_wip::Orbit orbit (refepoch, spacing, size);

    EXPECT_EQ( orbit.refepoch(), refepoch );
    EXPECT_EQ( orbit.spacing(), spacing );
    EXPECT_EQ( orbit.size(), size );
}

TEST_F(OrbitTest, CopyConstructorFromHost)
{
    isce::orbit_wip::Orbit h_orbit (refepoch, spacing, size);

    isce::cuda::orbit_wip::Orbit d_orbit (h_orbit);

    EXPECT_EQ( d_orbit.refepoch(), h_orbit.refepoch() );
    EXPECT_EQ( d_orbit.spacing(), h_orbit.spacing() );
    EXPECT_EQ( d_orbit.size(), h_orbit.size() );
}

TEST_F(OrbitTest, AssignFromHost)
{
    isce::orbit_wip::Orbit h_orbit (refepoch, spacing, size);

    isce::cuda::orbit_wip::Orbit d_orbit;
    d_orbit = h_orbit;

    EXPECT_EQ( d_orbit.refepoch(), h_orbit.refepoch() );
    EXPECT_EQ( d_orbit.spacing(), h_orbit.spacing() );
    EXPECT_EQ( d_orbit.size(), h_orbit.size() );
}

TEST_F(OrbitTest, CopyToHost)
{
    isce::cuda::orbit_wip::Orbit d_orbit (refepoch, spacing, size);

    isce::orbit_wip::Orbit h_orbit = d_orbit;

    EXPECT_EQ( h_orbit.refepoch(), d_orbit.refepoch() );
    EXPECT_EQ( h_orbit.spacing(), d_orbit.spacing() );
    EXPECT_EQ( h_orbit.size(), d_orbit.size() );
}

TEST_F(OrbitTest, OrbitView)
{
    typedef isce::cuda::orbit_wip::Orbit Orbit;

    Orbit orbit = Orbit::from_statevectors(statevecs);

    isce::cuda::orbit_wip::OrbitView view = orbit;

    EXPECT_EQ( orbit.refepoch(), view.refepoch() );
    EXPECT_EQ( orbit.time(), view.time() );
    EXPECT_EQ( orbit.position().data().get(), view.position() );
    EXPECT_EQ( orbit.velocity().data().get(), view.velocity() );
}

TEST_F(OrbitTest, Accessor)
{
    isce::cuda::orbit_wip::Orbit orbit (refepoch, spacing, size);

    orbit[0] = statevecs[0];
    orbit[1] = statevecs[1];

    EXPECT_EQ( orbit[0], statevecs[0] );
    EXPECT_EQ( orbit[1], statevecs[1] );
}

TEST_F(OrbitTest, ConstAccessor)
{
    typedef isce::cuda::orbit_wip::Orbit Orbit;

    const Orbit orbit = Orbit::from_statevectors(statevecs);

    EXPECT_EQ( orbit[0], statevecs[0] );
    EXPECT_EQ( orbit[1], statevecs[1] );
}

TEST_F(OrbitTest, AccessorDateTimeMismatch)
{
    typedef isce::cuda::orbit_wip::Orbit Orbit;

    Orbit orbit = Orbit::from_statevectors(statevecs);

    isce::core::StateVector statevec = statevecs[0];
    statevec.datetime += 1.;

    // datetime of statevector must be = refepoch() + spacing() * idx
    EXPECT_THROW( orbit[0] = statevec, isce::except::InvalidArgument );
}

TEST_F(OrbitTest, PushBack)
{
    isce::cuda::orbit_wip::Orbit orbit (refepoch, spacing);

    orbit.push_back(statevecs[0]);
    orbit.push_back(statevecs[1]);

    {
        EXPECT_EQ( orbit.time()[0], 0. );
        isce::core::Vec3 pos = orbit.position()[0];
        EXPECT_EQ( pos, statevecs[0].position );
        isce::core::Vec3 vel = orbit.velocity()[0];
        EXPECT_EQ( vel, statevecs[0].velocity );
    }

    {
        EXPECT_EQ( orbit.time()[1], spacing.getTotalSeconds() );
        isce::core::Vec3 pos = orbit.position()[1];
        EXPECT_EQ( pos, statevecs[1].position );
        isce::core::Vec3 vel = orbit.velocity()[1];
        EXPECT_EQ( vel, statevecs[1].velocity );
    }
}

TEST_F(OrbitTest, PushBackDateTimeMismatch)
{
    isce::cuda::orbit_wip::Orbit orbit (refepoch, spacing);

    isce::core::StateVector statevec = statevecs[0];
    statevec.datetime += 1.;

    // datetime of next statevector must be = refepoch() + spacing() * size()
    EXPECT_THROW( orbit.push_back(statevec), isce::except::InvalidArgument );
}

TEST_F(OrbitTest, Resize)
{
    isce::cuda::orbit_wip::Orbit orbit (refepoch, spacing, size);

    int new_size = 5;
    orbit.resize(new_size);

    EXPECT_EQ( orbit.size(), new_size );
}

TEST_F(OrbitTest, Empty)
{
    {
        isce::cuda::orbit_wip::Orbit orbit (refepoch, spacing, 0);
        EXPECT_TRUE( orbit.empty() );
    }

    {
        isce::cuda::orbit_wip::Orbit orbit (refepoch, spacing, size);
        EXPECT_FALSE( orbit.empty() );
    }
}

TEST_F(OrbitTest, ToStateVectors)
{
    isce::cuda::orbit_wip::Orbit orbit (refepoch, spacing);

    orbit.push_back(statevecs[0]);
    orbit.push_back(statevecs[1]);

    std::vector<isce::core::StateVector> orbit_statevecs = orbit.to_statevectors();

    EXPECT_EQ( orbit_statevecs, statevecs );
}

TEST_F(OrbitTest, Subinterval)
{
    isce::cuda::orbit_wip::Orbit orbit (refepoch, spacing);

    orbit.push_back(statevecs[0]);
    orbit.push_back(statevecs[1]);

    int start = 1;
    int stop = 2;

    isce::cuda::orbit_wip::OrbitView view = orbit.subinterval(start, stop);

    EXPECT_EQ( view.refepoch(), orbit.refepoch() + orbit.spacing() * start );
    EXPECT_EQ( view.spacing(), orbit.spacing().getTotalSeconds() );
    EXPECT_EQ( view.size(), stop - start );
    EXPECT_EQ( view.position(), (&orbit.position()[start]).get() );
    EXPECT_EQ( view.velocity(), (&orbit.velocity()[start]).get() );
}

TEST_F(OrbitTest, Comparison)
{
    typedef isce::cuda::orbit_wip::Orbit Orbit;

    Orbit orbit1 = Orbit::from_statevectors(statevecs);
    Orbit orbit2 = Orbit::from_statevectors(statevecs);
    Orbit orbit3;

    EXPECT_TRUE( orbit1 == orbit2 );
    EXPECT_TRUE( orbit1 != orbit3 );
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

