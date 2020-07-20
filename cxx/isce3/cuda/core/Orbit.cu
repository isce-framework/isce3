#include "Orbit.h"

#include <isce3/core/Common.h>
#include <isce3/core/detail/BuildOrbit.h>
#include <isce3/error/ErrorCode.h>
#include <isce3/except/Error.h>

#include <isce3/cuda/except/Error.h>

#include "OrbitView.h"

using isce3::core::DateTime;
using isce3::core::OrbitInterpMethod;
using isce3::core::OrbitInterpBorderMode;
using isce3::core::StateVector;
using isce3::core::TimeDelta;
using isce3::core::Vec3;
using isce3::error::ErrorCode;
using isce3::error::getErrorString;

using HostOrbit = isce3::core::Orbit;

namespace isce3 { namespace cuda { namespace core {

Orbit::Orbit(const HostOrbit & orbit)
:
    _reference_epoch(orbit.referenceEpoch()),
    _time(orbit.time()),
    _position(orbit.position()),
    _velocity(orbit.velocity()),
    _interp_method(orbit.interpMethod())
{}

Orbit::Orbit(const std::vector<StateVector> & statevecs,
             OrbitInterpMethod interp_method)
:
    Orbit(HostOrbit(statevecs, interp_method))
{}

Orbit::Orbit(const std::vector<StateVector> & statevecs,
             const DateTime & reference_epoch,
             OrbitInterpMethod interp_method)
:
    Orbit(HostOrbit(statevecs, reference_epoch, interp_method))
{}

// convenience function to get device vector data pointer
template<typename T>
constexpr
const T * dptr(const thrust::device_vector<T> & v) { return v.data().get(); }

// convenience function to get device vector data pointer
template<typename T>
constexpr
T * dptr(thrust::device_vector<T> & v) { return v.data().get(); }

// copy device vector to std::vector
template<typename T>
inline
std::vector<T> copyToHost(const thrust::device_vector<T> & d)
{
    std::vector<T> h(d.size());

    if (d.size() != 0) {
        T * dst = h.data();
        const T * src = dptr(d);
        std::size_t count = d.size() * sizeof(T);
        checkCudaErrors( cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost) );
    }

    return h;
}

std::vector<StateVector> Orbit::getStateVectors() const
{
    // copy to host
    std::vector<Vec3> pos = copyToHost(_position);
    std::vector<Vec3> vel = copyToHost(_velocity);

    // convert to state vectors
    std::vector<StateVector> statevecs(size());
    for (int i = 0; i < size(); ++i) {
        statevecs[i].datetime = _reference_epoch + TimeDelta(_time[i]);
        statevecs[i].position = _position[i];
        statevecs[i].velocity = _velocity[i];
    }

    return statevecs;
}

void Orbit::setStateVectors(const std::vector<StateVector> & statevecs)
{
    _time = isce3::core::detail::getOrbitTime(statevecs, _reference_epoch);
    _position = isce3::core::detail::getOrbitPosition(statevecs);
    _velocity = isce3::core::detail::getOrbitVelocity(statevecs);
}

void Orbit::referenceEpoch(const DateTime & reference_epoch)
{
    DateTime old_refepoch = _reference_epoch;
    double old_starttime = _time.first();

    double dt = (old_refepoch - reference_epoch).getTotalSeconds();
    double starttime = old_starttime + dt;

    _time.first(starttime);
    _reference_epoch = reference_epoch;
}

static
CUDA_GLOBAL
void interpOrbit(Vec3 * position,
                 Vec3 * velocity,
                 OrbitView orbit,
                 double t,
                 OrbitInterpBorderMode border_mode,
                 ErrorCode * status)
{
    // no bounds checking - assume single-threaded execution
    ErrorCode ret = orbit.interpolate(position, velocity, t, border_mode);
    if (status && ret != ErrorCode::Success) {
        *status = ret;
    }
}

ErrorCode Orbit::interpolate(Vec3 * position,
                        Vec3 * velocity,
                        double t,
                        OrbitInterpBorderMode border_mode) const
{
    // init device memory for results & status code
    thrust::device_vector<Vec3> d_pos(1), d_vel(1);
    thrust::device_vector<ErrorCode> d_stat(1, ErrorCode::Success);

    // launch kernel, check for launch & execution errors
    interpOrbit<<<1, 1>>>(dptr(d_pos), dptr(d_vel), *this, t, border_mode, dptr(d_stat));
    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaStreamSynchronize(0) );

    // check return code
    ErrorCode status = d_stat[0];
    if (status != ErrorCode::Success and
            border_mode == OrbitInterpBorderMode::Error) {

        std::string errmsg = getErrorString(status);
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errmsg);
    }

    if (position) { *position = d_pos[0]; }
    if (velocity) { *velocity = d_vel[0]; }

    return status;
}

bool operator==(const Orbit & lhs, const Orbit & rhs)
{
    return lhs.referenceEpoch() == rhs.referenceEpoch() &&
           lhs.time() == rhs.time() &&
           lhs.position() == rhs.position() &&
           lhs.velocity() == rhs.velocity() &&
           lhs.interpMethod() == rhs.interpMethod();
}

bool operator!=(const Orbit & lhs, const Orbit & rhs)
{
    return !(lhs == rhs);
}

}}}
