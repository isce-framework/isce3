#include <isce/cuda/except/Error.h>
#include <isce/cuda/orbit_wip/Orbit.h>
#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace isce::core;
using namespace isce::cuda::orbit_wip;

bool isclose(const Vec3 & lhs, const Vec3 & rhs)
{
    double errtol = 1e-6;
    return std::abs(lhs[0] - rhs[0]) < errtol &&
           std::abs(lhs[1] - rhs[1]) < errtol &&
           std::abs(lhs[2] - rhs[2]) < errtol;
}

std::ostream & operator<<(std::ostream & os, const Vec3 & v)
{
    return os << std::endl << "{ " << v[0] << ", " << v[1] << ", " << v[2] << " }";
}

__global__
void interp_hermite(Vec3 * position,
        Vec3 * velocity,
        isce::cuda::orbit_wip::OrbitView orbit,
        const double * time,
        int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size) {
        return;
    }

    hermite_interpolate(orbit, time[tid], &position[tid], &velocity[tid]);
}

__global__
void interp_legendre(Vec3 * position,
        Vec3 * velocity,
        isce::cuda::orbit_wip::OrbitView orbit,
        const double * time,
        int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size) {
        return;
    }

    legendre_interpolate(orbit, time[tid], &position[tid], &velocity[tid]);
}

__global__
void interp_sch(Vec3 * position,
        Vec3 * velocity,
        isce::cuda::orbit_wip::OrbitView orbit,
        const double * time,
        int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size) {
        return;
    }

    sch_interpolate(orbit, time[tid], &position[tid], &velocity[tid]);
}

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
    thrust::host_vector<double> test_times;
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

        test_times = std::vector<double> {23.3, 36.7, 54.5, 89.3};

        for (size_t i = 0; i < test_times.size(); ++i) {
            TimeDelta dt = test_times[i];
            StateVector sv = make_linear_statevec(refepoch, dt, initial_pos, velocity);
            expected.push_back(sv);
        }
    }
};

TEST_F(InterpOrbitTest_Linear, HermiteInterpolate)
{
    // copy test times to device
    thrust::device_vector<double> d_time = test_times;

    // output interpolated position and velocity
    thrust::device_vector<Vec3> pos (test_times.size());
    thrust::device_vector<Vec3> vel (test_times.size());

    Vec3 * _pos = pos.data().get();
    Vec3 * _vel = vel.data().get();
    double * _time = d_time.data().get();
    int ntest = test_times.size();

    // interpolate
    int block = 128;
    int grid = (ntest + block - 1) / block;
    interp_hermite<<<grid, block>>>(_pos, _vel, orbit, _time, ntest);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // copy results to host
    thrust::host_vector<Vec3> h_pos = pos;
    thrust::host_vector<Vec3> h_vel = vel;

    // check results
    for (int i = 0; i < ntest; ++i) {
        EXPECT_PRED2( isclose, h_pos[i], expected[i].position );
        EXPECT_PRED2( isclose, h_vel[i], expected[i].velocity );
    }
}

TEST_F(InterpOrbitTest_Linear, LegendreInterpolate)
{
    // copy test times to device
    thrust::device_vector<double> d_time = test_times;

    // output interpolated position and velocity
    thrust::device_vector<Vec3> pos (test_times.size());
    thrust::device_vector<Vec3> vel (test_times.size());

    Vec3 * _pos = pos.data().get();
    Vec3 * _vel = vel.data().get();
    double * _time = d_time.data().get();
    int ntest = test_times.size();

    // interpolate
    int block = 128;
    int grid = (ntest + block - 1) / block;
    interp_legendre<<<grid, block>>>(_pos, _vel, orbit, _time, ntest);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // copy results to host
    thrust::host_vector<Vec3> h_pos = pos;
    thrust::host_vector<Vec3> h_vel = vel;

    // check results
    for (int i = 0; i < ntest; ++i) {
        EXPECT_PRED2( isclose, h_pos[i], expected[i].position );
        EXPECT_PRED2( isclose, h_vel[i], expected[i].velocity );
    }
}

TEST_F(InterpOrbitTest_Linear, SCHInterpolate)
{
    // copy test times to device
    thrust::device_vector<double> d_time = test_times;

    // output interpolated position and velocity
    thrust::device_vector<Vec3> pos (test_times.size());
    thrust::device_vector<Vec3> vel (test_times.size());

    Vec3 * _pos = pos.data().get();
    Vec3 * _vel = vel.data().get();
    double * _time = d_time.data().get();
    int ntest = test_times.size();

    // interpolate
    int block = 128;
    int grid = (ntest + block - 1) / block;
    interp_sch<<<grid, block>>>(_pos, _vel, orbit, _time, ntest);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // copy results to host
    thrust::host_vector<Vec3> h_pos = pos;
    thrust::host_vector<Vec3> h_vel = vel;

    // check results
    for (int i = 0; i < ntest; ++i) {
        EXPECT_PRED2( isclose, h_pos[i], expected[i].position );
        EXPECT_PRED2( isclose, h_vel[i], expected[i].velocity );
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
    thrust::host_vector<double> test_times;
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

        test_times = std::vector<double> {23.3, 36.7, 54.5, 89.3};

        for (size_t i = 0; i < test_times.size(); ++i) {
            TimeDelta dt = test_times[i];
            StateVector sv = make_polynomial_statevec(refepoch, dt, coeffs);
            expected.push_back(sv);
        }
    }
};

TEST_F(InterpOrbitTest_Polynomial, HermiteInterpolate)
{
    // copy test times to device
    thrust::device_vector<double> d_time = test_times;

    // output interpolated position and velocity
    thrust::device_vector<Vec3> pos (test_times.size());
    thrust::device_vector<Vec3> vel (test_times.size());

    Vec3 * _pos = pos.data().get();
    Vec3 * _vel = vel.data().get();
    double * _time = d_time.data().get();
    int ntest = test_times.size();

    // interpolate
    int block = 128;
    int grid = (ntest + block - 1) / block;
    interp_hermite<<<grid, block>>>(_pos, _vel, orbit, _time, ntest);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // copy results to host
    thrust::host_vector<Vec3> h_pos = pos;
    thrust::host_vector<Vec3> h_vel = vel;

    // check results
    for (int i = 0; i < ntest; ++i) {
        EXPECT_PRED2( isclose, h_pos[i], expected[i].position );
        EXPECT_PRED2( isclose, h_vel[i], expected[i].velocity );
    }
}

TEST_F(InterpOrbitTest_Polynomial, LegendreInterpolate)
{
    // copy test times to device
    thrust::device_vector<double> d_time = test_times;

    // output interpolated position and velocity
    thrust::device_vector<Vec3> pos (test_times.size());
    thrust::device_vector<Vec3> vel (test_times.size());

    Vec3 * _pos = pos.data().get();
    Vec3 * _vel = vel.data().get();
    double * _time = d_time.data().get();
    int ntest = test_times.size();

    // interpolate
    int block = 128;
    int grid = (ntest + block - 1) / block;
    interp_legendre<<<grid, block>>>(_pos, _vel, orbit, _time, ntest);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // copy results to host
    thrust::host_vector<Vec3> h_pos = pos;
    thrust::host_vector<Vec3> h_vel = vel;

    // check results
    for (int i = 0; i < ntest; ++i) {
        EXPECT_PRED2( isclose, h_pos[i], expected[i].position );
        EXPECT_PRED2( isclose, h_vel[i], expected[i].velocity );
    }
}

TEST_F(InterpOrbitTest_Polynomial, SCHInterpolate)
{
    // copy test times to device
    thrust::device_vector<double> d_time = test_times;

    // output interpolated position and velocity
    thrust::device_vector<Vec3> pos (test_times.size());
    thrust::device_vector<Vec3> vel (test_times.size());

    Vec3 * _pos = pos.data().get();
    Vec3 * _vel = vel.data().get();
    double * _time = d_time.data().get();
    int ntest = test_times.size();

    // interpolate
    int block = 128;
    int grid = (ntest + block - 1) / block;
    interp_sch<<<grid, block>>>(_pos, _vel, orbit, _time, ntest);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // copy results to host
    thrust::host_vector<Vec3> h_pos = pos;
    thrust::host_vector<Vec3> h_vel = vel;

    // check results
    for (int i = 0; i < ntest; ++i) {
        EXPECT_PRED2( isclose, h_pos[i], expected[i].position );
        EXPECT_PRED2( isclose, h_vel[i], expected[i].velocity );
    }
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

