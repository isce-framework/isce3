#include "Backproject.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>
#include <limits>
#include <pyre/journal.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <typeinfo>

#include <isce3/container/RadarGeometry.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Linspace.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/Projections.h>
#include <isce3/core/Vector.h>
#include <isce3/cuda/container/RadarGeometry.h>
#include <isce3/cuda/core/Interp1d.h>
#include <isce3/cuda/core/Kernels.h>
#include <isce3/cuda/core/Orbit.h>
#include <isce3/cuda/core/OrbitView.h>
#include <isce3/cuda/core/gpuLUT2d.h>
#include <isce3/cuda/except/Error.h>
#include <isce3/cuda/fft/FFT.h>
#include <isce3/cuda/geometry/gpuDEMInterpolator.h>
#include <isce3/cuda/geometry/gpuGeometry.h>
#include <isce3/cuda/signal/NFFT2d.h>
#include <isce3/fft/FFT.h>
#include <isce3/fft/FFTUtil.h>
#include <isce3/focus/BistaticDelay.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/geometry.h>
#include <isce3/geometry/rdr2geo_roots.h>
#include <isce3/geometry/geo2rdr_roots.h>

using namespace isce3::core;
using namespace isce3::cuda::geometry;

using isce3::cuda::core::interp1d;
using isce3::error::ErrorCode;
using isce3::focus::bistaticDelay;
using isce3::focus::dryTropoDelayTSX;
using isce3::focus::PolarGrid;
using isce3::focus::setupPolarGridForPulses;
using isce3::cuda::signal::NFFT2dResult;
using isce3::cuda::signal::NFFT2dResultView;

using HostDEMInterpolator = isce3::geometry::DEMInterpolator;
using HostRadarGeometry = isce3::container::RadarGeometry;

using DeviceDEMInterpolator = isce3::cuda::geometry::gpuDEMInterpolator;
using DeviceOrbitView = isce3::cuda::core::OrbitView;
using DeviceRadarGeometry = isce3::cuda::container::RadarGeometry;

template<typename T>
using DeviceLUT2d = isce3::cuda::core::gpuLUT2d<T>;

// clang-format off
template<typename T> using HostBartlettKernel = isce3::core::BartlettKernel<T>;
template<typename T> using HostChebyKernel = isce3::core::ChebyKernel<T>;
template<typename T> using HostKnabKernel = isce3::core::KnabKernel<T>;
template<typename T> using HostLinearKernel = isce3::core::LinearKernel<T>;
template<typename T> using HostTabulatedKernel = isce3::core::TabulatedKernel<T>;

template<typename T> using DeviceBartlettKernel = isce3::cuda::core::BartlettKernel<T>;
template<typename T> using DeviceChebyKernel = isce3::cuda::core::ChebyKernel<T>;
template<typename T> using DeviceKnabKernel = isce3::cuda::core::KnabKernel<T>;
template<typename T> using DeviceLinearKernel = isce3::cuda::core::LinearKernel<T>;
template<typename T> using DeviceTabulatedKernel = isce3::cuda::core::TabulatedKernel<T>;
// clang-format on

namespace isce3 { namespace cuda { namespace focus {

/**
 * \internal
 * Interpolate platform position and velocity at a range of uniformly-spaced
 * timepoints.
 *
 * The global error code is set if any thread encounters an error.
 *
 * \param[out] pos   Interpolated positions (m)
 * \param[out] vel   Interpolated velocities (m/s)
 * \param[in]  orbit Platform orbit
 * \param[in]  t     Interpolation times w.r.t. reference epoch (s)
 * \param[out] errc  Error flag
 */
__global__ void interpolateOrbit(Vec3* pos, Vec3* vel,
                                 const DeviceOrbitView orbit,
                                 const Linspace<double> t, ErrorCode* errc)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    // bounds check
    if (tid >= t.size()) {
        return;
    }

    // interpolate orbit
    const auto status = orbit.interpolate(&pos[tid], &vel[tid], t[tid]);

    // check error code
    if (status != ErrorCode::Success) {
        *errc = status;
    }
}

/**
 * \internal
 * Interpolate platform position and velocity at a series of timepoints.
 *
 * The global error code is set if any thread encounters an error.
 *
 * \param[out] pos   Interpolated positions (m)
 * \param[out] vel   Interpolated velocities (m/s)
 * \param[in]  orbit Platform orbit
 * \param[in]  t     Interpolation times w.r.t. reference epoch (s)
 * \param[in]  n     Number of timepoints
 * \param[out] errc  Error flag
 */
__global__ void interpolateOrbit(Vec3* pos, Vec3* vel,
                                 const DeviceOrbitView orbit, const double* t,
                                 const size_t n, ErrorCode* errc)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    if (tid >= n) {
        return;
    }

    // interpolate orbit
    const auto status = orbit.interpolate(&pos[tid], &vel[tid], t[tid]);

    // check error code
    if (status != ErrorCode::Success) {
        *errc = status;
    }
}

/**
 * \internal
 * Transform a 2D radar grid from radar coordinates (azimuth, range) to
 * geodetic coordinates (longitude, latitude, height).
 *
 * The radar grid is defined by the \p azimuth_time and \p slant_range inputs.
 *
 * The global error code is set if any thread encounters an error.
 *
 * \param[out] xyz_out      ECEF XYZ of each target (m)
 * \param[in]  azimuth_time Azimuth time coordinates w.r.t. reference epoch (s)
 * \param[in]  slant_range  Slant range coordinates (m)
 * \param[in]  doppler      Doppler model
 * \param[in]  orbit        Platform orbit
 * \param[in]  dem          DEM sampling interface
 * \param[in]  ellipsoid    Reference ellipsoid
 * \param[in]  wvl          Radar wavelength (m)
 * \param[in]  side         Radar look side
 * \param[in]  params       Root-finding algorithm parameters
 * \param[out] errc         Error flag
 */
__global__ void runRdr2Geo(Vec3* xyz_out, const Linspace<double> azimuth_time,
                           const Linspace<double> slant_range,
                           const DeviceLUT2d<double> doppler,
                           const DeviceOrbitView orbit,
                           DeviceDEMInterpolator dem, const Ellipsoid ellipsoid,
                           const double wvl, const LookSide side,
                           const Rdr2GeoBracketParams params,
                           ErrorCode* errc)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    const auto lines = static_cast<size_t>(azimuth_time.size());
    const auto samples = static_cast<size_t>(slant_range.size());
    if (tid >= lines * samples) {
        return;
    }

    // convert flat index to 2D array indices
    const auto j = static_cast<int>(tid / samples);
    const auto i = static_cast<int>(tid % samples);

    // evaluate Doppler model at target position
    const double t = azimuth_time[j];
    const double r = slant_range[i];
    const double fd = doppler.eval(t, r);

    // make thread-local variable to store rdr2geo output
    Vec3 xyz;

    const int converged =
            rdr2geo_bracket(t, r, fd, orbit, ellipsoid, dem, xyz, wvl, side,
                    params.tol_height, params.look_min, params.look_max);

    // check convergence
    if (converged) {
        xyz_out[tid] = xyz;
    } else {
        // set output to NaN (?)
        constexpr static auto nan = std::numeric_limits<double>::quiet_NaN();
        xyz_out[tid] = {nan, nan, nan};

        // set global error flag
        *errc = ErrorCode::FailedToConverge;
    }
}

/**
 * \internal
 * Transform each input target position from geodetic coordinates (longitude,
 * latitude, height) to radar coordinates (azimuth, range).
 *
 * The global error code is set if any thread encounters an error.
 *
 * \param[out] t_out        Azim. time of each target w.r.t. reference epoch (s)
 * \param[out] r_out        Slant range of each target (m)
 * \param[in]  xyz_in       ECEF XYZ of each target (m)
 * \param[in]  n            Number of targets
 * \param[in]  orbit        Platform orbit
 * \param[in]  doppler      Doppler model
 * \param[in]  wvl          Radar wavelength (m)
 * \param[in]  side         Radar look side
 * \param[in]  params       Root-finding algorithm parameters
 * \param[out] errc         Error flag
 */
__global__ void runGeo2Rdr(double* t_out, double* r_out, const Vec3* xyz_in,
                           const size_t n,
                           const DeviceOrbitView orbit,
                           const DeviceLUT2d<double> doppler, const double wvl,
                           const LookSide side,
                           const Geo2RdrBracketParams params, ErrorCode* errc)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    if (tid >= n) {
        return;
    }

    // make thread-local variables to store geo2rdr output range, azimuth time
    double t, r;

    // run geo2rdr
    auto xyz = xyz_in[tid];
    int converged =
            geo2rdr_bracket(xyz, orbit, doppler, &t, &r, wvl, side,
                    params.tol_aztime, params.time_start, params.time_end);

    // check convergence
    if (converged) {
        t_out[tid] = t;
        r_out[tid] = r;
    } else {
        // set outputs to NaN (?)
        constexpr static auto nan = std::numeric_limits<double>::quiet_NaN();
        t_out[tid] = nan;
        r_out[tid] = nan;

        // set global error flag
        *errc = ErrorCode::FailedToConverge;
    }
}

/**
 * \internal
 * Transform target coordinates from ECEF to LLH, given some reference ellipsoid
 *
 * \param[out] llh       Lon/lat/hae coordinates of each target (deg/deg/m)
 * \param[in]  xyz       ECEF coordinates of each target (m)
 * \param[in]  n         Number of targets
 * \param[in]  ellipsoid Reference ellipsoid
 */
__global__ void ecef2llh(Vec3* llh, const Vec3* xyz, const size_t n,
                         const Ellipsoid ellipsoid)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    if (tid >= n) {
        return;
    }

    // transform coordinates from ECEF to LLH
    llh[tid] = ellipsoid.xyzToLonLat(xyz[tid]);
}

/**
 * \internal
 * Estimate dry troposphere delay for one or more targets using the TerraSAR-X
 * model.
 *
 * \param[out] tau_atm   Dry troposphere delay for each target (s)
 * \param[in]  p         Platform position at target's azimuth time (m)
 * \param[in]  llh       Lon/lat/hae coordinates of each target (deg/deg/m)
 * \param[in]  n         Number of targets
 * \param[in]  ellipsoid Reference ellipsoid
 */
__global__ void estimateDryTropoDelayTSX(double* tau_atm, const Vec3* p,
                                         const Vec3* llh, const size_t n,
                                         const Ellipsoid ellipsoid)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    if (tid >= n) {
        return;
    }

    // estimate dry troposphere delay
    tau_atm[tid] = dryTropoDelayTSX(p[tid], llh[tid], ellipsoid);
}

/**
 * \internal
 * Estimate coherent processing window bounds for one or more targets.
 *
 * Returns the indices of the first pulse and one past the last pulse to
 * coherently integrate for each target.
 *
 * \param[out] kstart_out   Processing window start pulse (inclusive)
 * \param[out] kstop_out    Processing window end pulse (exclusive)
 * \param[in]  t_in         Azim. time of each target w.r.t. reference epoch (s)
 * \param[in]  r_in         Slant range of each target (m)
 * \param[in]  x_in         Position of each target in ECEF coords (m)
 * \param[in]  p_in         Platform position at each target's azimuth time (m)
 * \param[in]  v_in         Platform velocity at each target's azimuth time (m)
 * \param[in]  n            Number of targets
 * \param[in]  azimuth_time Azim. time of each pulse w.r.t. reference epoch (s)
 * \param[in]  wvl          Radar wavelength (m)
 * \param[in]  ds           Desired azimuth resolution (m)
 */
__global__ void getCPIBounds(int* kstart_out, int* kstop_out,
                             const double* t_in, const double* r_in,
                             const Vec3* x_in, const Vec3* p_in,
                             const Vec3* v_in, const size_t n,
                             const Linspace<double> azimuth_time,
                             const double wvl, const double ds)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    if (tid >= n) {
        return;
    }

    // load inputs
    const double t = t_in[tid];
    const double r = r_in[tid];
    const Vec3 p = p_in[tid];
    const Vec3 v = v_in[tid];
    const Vec3 x = x_in[tid];

    // estimate synthetic aperture length required to achieve the desired
    // azimuth resolution
    const double l = wvl * r * (p.norm() / x.norm()) / (2. * ds);

    // approximate CPI duration (assuming constant platform velocity)
    const double cpi = l / v.norm();

    // get coherent processing window start & end time
    const double tstart = t - 0.5 * cpi;
    const double tstop = t + 0.5 * cpi;

    // convert CPI bounds to pulse indices
    const double t0 = azimuth_time.first();
    const double dt = azimuth_time.spacing();
    const auto kstart = static_cast<int>(std::floor((tstart - t0) / dt));
    const auto kstop = static_cast<int>(std::ceil((tstop - t0) / dt));

    kstart_out[tid] = std::max(kstart, 0);
    kstop_out[tid] = std::min(kstop, azimuth_time.size());
}

/**
 * \internal
 * Estimate coherent processing window bounds for one or more targets.
 *
 * Returns the indices of the first pulse and one past the last pulse to
 * coherently integrate for each target.
 *
 * \param[out] tstart_out   Processing window start time (inclusive)
 * \param[out] tstop_out    Processing window end time (exclusive)
 * \param[in]  t_in         Azim. time of each target w.r.t. reference epoch (s)
 * \param[in]  r_in         Slant range of each target (m)
 * \param[in]  x_in         Position of each target in ECEF coords (m)
 * \param[in]  p_in         Platform position at each target's azimuth time (m)
 * \param[in]  v_in         Platform velocity at each target's azimuth time (m)
 * \param[in]  n            Number of targets
 * \param[in]  wvl          Radar wavelength (m)
 * \param[in]  ds           Desired azimuth resolution (m)
 */
__global__ void getCPITimeBounds(double* tstart_out, double* tstop_out,
                             const double* t_in, const double* r_in,
                             const Vec3* x_in, const Vec3* p_in,
                             const Vec3* v_in, const size_t n,
                             const double wvl, const double ds)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    if (tid >= n) {
        return;
    }

    // load inputs
    const double t = t_in[tid];
    const double r = r_in[tid];
    const Vec3 p = p_in[tid];
    const Vec3 v = v_in[tid];
    const Vec3 x = x_in[tid];

    // estimate synthetic aperture length required to achieve the desired
    // azimuth resolution
    const double l = wvl * r * (p.norm() / x.norm()) / (2. * ds);

    // approximate CPI duration (assuming constant platform velocity)
    const double cpi = l / v.norm();

    // get coherent processing window start & end time
    tstart_out[tid] = t - 0.5 * cpi;
    tstop_out[tid] = t + 0.5 * cpi;
}

/**
 * \internal
 * Backprojection core processing loop
 *
 * Compress the radar return from each input target in azimuth by coherently
 * integrating the echos from a range of pulses.
 *
 * Operates on pulses from a batch of the full range-compressed swath
 * bounded by [ \p batch_start , \p batch_stop ). The result for each target is
 * added to the previous pixel value, which may contain partially-compressed
 * data from previous batches. Therefore, the output array should be initialized
 * to zero prior to batch processing.
 *
 * \param[in,out] out         Output focused image data for each target
 * \param[in] rc              Range-compressed signal data batch
 * \param[in] pos             Platform position at each pulse (m)
 * \param[in] vel             Platform velocity at each pulse (m/s)
 * \param[in] sampling_window Range sampling window (s)
 * \param[in] x_in            ECEF position of each target (m)
 * \param[in] tau_atm_in      Dry troposphere delay for each target (s)
 * \param[in] kstart_in       First pulse in CPI for each target
 * \param[in] kstop_in        One past the last pulse in the CPI for each target
 * \param[in] n               Number of targets
 * \param[in] fc              Center frequency (Hz)
 * \param[in] kernel          1D interpolation kernel
 * \param[in] batch_start     First pulse in batch of range-compressed data
 * \param[in] batch_stop      One past the last pulse in the rc data batch
 */
template<class Kernel>
__global__ void sumCoherentBatch(
        thrust::complex<float>* out, const thrust::complex<float>* rc,
        const Vec3* __restrict__ pos, const Vec3* __restrict__ vel,
        const Linspace<double> sampling_window, const Vec3* __restrict__ x_in,
        const double* __restrict__ tau_atm_in,
        const int* __restrict__ kstart_in, const int* __restrict__ kstop_in,
        const size_t n, const double fc, const Kernel kernel,
        const int batch_start, const int batch_stop)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    if (tid >= n) {
        return;
    }

    // cache some inputs
    const Vec3 x = x_in[tid];
    const double tau_atm = tau_atm_in[tid];
    const int kstart = kstart_in[tid];
    const int kstop = kstop_in[tid];

    // set bad data to NaN
    if (std::isnan(x[0])) {
        const auto nan = std::nanf("geometry");
        out[tid] = thrust::complex<float>(nan, nan);
        return;
    }

    // get range sampling window start, spacing, number of samples
    const double tau0 = sampling_window.first();
    const double dtau = sampling_window.spacing();
    const auto samples = static_cast<size_t>(sampling_window.size());

    // loop over lines in batch
    thrust::complex<double> batch_sum = {0., 0.};
    for (int k = batch_start; k < batch_stop; ++k) {

        // check if pulse is within CPI bounds
        if (k < kstart or k >= kstop) {
            continue;
        }

        // compute round-trip delay to target
        const double tau = tau_atm + bistaticDelay(pos[k], vel[k], x);

        // interpolate range-compressed data
        const auto* rc_line = &rc[(k - batch_start) * samples];
        const double u = (tau - tau0) / dtau;
        thrust::complex<double> z = interp1d(kernel, rc_line, samples, 1, u);

        // apply phase migration compensation
        double sin_phi, cos_phi;
        ::sincospi(2. * fc * tau, &sin_phi, &cos_phi);
        z *= thrust::complex<double>(cos_phi, sin_phi);

        batch_sum += z;
    }

    // add batch sum to total
    out[tid] += thrust::complex<float>(batch_sum);
}

template <typename T>
__global__ void broadcastMultiply(const T* row, size_t ncol, T* image, size_t npix)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    if (tid >= npix) {
        return;
    }

    auto i_col = tid % ncol;

    image[tid] *= row[i_col];
}

__global__ void
makeSubApertureMask(
    const double subaperture_start, const double subaperture_end,
    const size_t n,
    const double* pixel_start,
    const double* pixel_end,
    bool* mask)
{
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    if (tid >= n) {
        return;
    }

    mask[tid] = (subaperture_end > pixel_start[tid])
            and (subaperture_start < pixel_end[tid]);
}


template<class Kernel>
ErrorCode backproject(std::complex<float>* out,
                      const DeviceRadarGeometry& out_geometry,
                      const std::complex<float>* in,
                      const DeviceRadarGeometry& in_geometry,
                      DeviceDEMInterpolator& dem, double fc, double ds,
                      const Kernel& kernel, DryTroposphereModel dry_tropo_model,
                      const Rdr2GeoBracketParams& rdr2geo_params,
                      const Geo2RdrBracketParams& geo2rdr_params, int batch,
                      float* height)
{
    // XXX input reference epoch must match output reference epoch
    if (out_geometry.referenceEpoch() != in_geometry.referenceEpoch()) {
        std::string errmsg = "input reference epoch must match output "
                             "reference epoch";
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), errmsg);
    }

    // init device variable to return error codes from device code
    thrust::device_vector<ErrorCode> errc(1, ErrorCode::Success);

    // get input & output radar grid azimuth time & slant range coordinates
    const Linspace<double> in_azimuth_time = in_geometry.sensingTime();
    const Linspace<double> in_slant_range = in_geometry.slantRange();
    const Linspace<double> out_azimuth_time = out_geometry.sensingTime();
    const Linspace<double> out_slant_range = out_geometry.slantRange();

    // interpolate platform position & velocity at each pulse
    int in_lines = in_azimuth_time.size();
    thrust::device_vector<Vec3> pos(in_lines);
    thrust::device_vector<Vec3> vel(in_lines);

    {
        const unsigned block = 256;
        const unsigned grid = (in_lines + block - 1) / block;

        interpolateOrbit<<<grid, block>>>(pos.data().get(), vel.data().get(),
                                          in_geometry.orbit(), in_azimuth_time,
                                          errc.data().get());

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    // range sampling window
    static constexpr double c = isce3::core::speed_of_light;
    const double swst = 2. * in_slant_range.first() / c;
    const double dtau = 2. * in_slant_range.spacing() / c;
    const int in_samples = in_slant_range.size();
    const auto sampling_window = Linspace<double>(swst, dtau, in_samples);

    // reference ellipsoid
    const int epsg = dem.epsgCode();
    const Ellipsoid ellipsoid = makeProjection(epsg)->ellipsoid();

    // carrier wavelength
    const double wvl = c / fc;

    // run rdr2geo using output geometry to get XYZ position of each target in
    // output grid
    const size_t out_grid_size =
            out_geometry.gridLength() * out_geometry.gridWidth();
    thrust::device_vector<Vec3> x(out_grid_size);

    {
        const unsigned block = 256;
        const unsigned grid = (out_grid_size + block - 1) / block;

        runRdr2Geo<<<grid, block>>>(x.data().get(), out_azimuth_time,
                                    out_slant_range, out_geometry.doppler(),
                                    out_geometry.orbit(), dem, ellipsoid, wvl,
                                    out_geometry.lookSide(), rdr2geo_params,
                                    errc.data().get());

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    // transform each target position from ECEF to LLH coordinates
    // NOTE only really needed if dumping height layer or doing TSX atmosphere
    // correction, but just compute it unconditionally.
    thrust::device_vector<Vec3> llh(out_grid_size);

    {
        const unsigned block = 256;
        const unsigned grid = (out_grid_size + block - 1) / block;

        ecef2llh<<<grid, block>>>(llh.data().get(), x.data().get(),
                                  out_grid_size, ellipsoid);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    if (height != nullptr) {
        thrust::device_vector<float> d_height(out_grid_size);
        thrust::transform(llh.begin(), llh.end(), d_height.begin(),
                [] __device__ (const Vec3& x) { return (float)x[2]; });
        checkCudaErrors(cudaMemcpy(height, d_height.data().get(),
                out_grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // run geo2rdr using input geometry to estimate the center of the coherent
    // processing window for each target
    thrust::device_vector<double> t(out_grid_size);
    thrust::device_vector<double> r(out_grid_size);

    {
        const unsigned block = 256;
        const unsigned grid = (out_grid_size + block - 1) / block;

        runGeo2Rdr<<<grid, block>>>(
                t.data().get(), r.data().get(), x.data().get(), out_grid_size,
                in_geometry.orbit(), in_geometry.doppler(), wvl,
                in_geometry.lookSide(), geo2rdr_params, errc.data().get());

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    // get platform position & velocity at center of CPI for each target
    thrust::device_vector<Vec3> p(out_grid_size);
    thrust::device_vector<Vec3> v(out_grid_size);

    {
        const unsigned block = 256;
        const unsigned grid = (out_grid_size + block - 1) / block;

        interpolateOrbit<<<grid, block>>>(p.data().get(), v.data().get(),
                                          in_geometry.orbit(), t.data().get(),
                                          out_grid_size, errc.data().get());

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    // estimate dry troposphere delay
    thrust::device_vector<double> tau_atm(out_grid_size);

    if (dry_tropo_model == DryTroposphereModel::NoDelay) {
        checkCudaErrors(cudaMemset(tau_atm.data().get(), 0,
                                   out_grid_size * sizeof(double)));
    } else if (dry_tropo_model == DryTroposphereModel::TSX) {
        const unsigned block = 256;
        const unsigned grid = (out_grid_size + block - 1) / block;

        estimateDryTropoDelayTSX<<<grid, block>>>(
                tau_atm.data().get(), p.data().get(), llh.data().get(),
                out_grid_size, ellipsoid);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    } else {
        std::string errmsg = "unexpected dry troposphere model";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    // get coherent integration bounds (pulse indices) for each target
    thrust::device_vector<int> kstart(out_grid_size);
    thrust::device_vector<int> kstop(out_grid_size);

    {
        const unsigned block = 256;
        const unsigned grid = (out_grid_size + block - 1) / block;

        getCPIBounds<<<grid, block>>>(
                kstart.data().get(), kstop.data().get(), t.data().get(),
                r.data().get(), x.data().get(), p.data().get(), v.data().get(),
                out_grid_size, in_azimuth_time, wvl, ds);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    // init device buffer for output focused image data
    thrust::device_vector<thrust::complex<float>> img(out_grid_size);

    // XXX not sure if img data is default initialized to zero
    checkCudaErrors(cudaMemset(img.data().get(), 0,
                               img.size() * sizeof(thrust::complex<float>)));

    // the full range-compressed swath may exceed device memory limitations
    // so we process out-of-core in batches of range-compressed pulses
    const size_t width = in_geometry.gridWidth();
    thrust::device_vector<thrust::complex<float>> rc(batch * width);

    // get the min & max processing window bounds from among all targets
    const int kstart_min = thrust::reduce(kstart.begin(), kstart.end(),
                                          std::numeric_limits<int>::max(),
                                          thrust::minimum<int>());
    const int kstop_max = thrust::reduce(kstop.begin(), kstop.end(),
                                         std::numeric_limits<int>::min(),
                                         thrust::maximum<int>());

    // iterate over batches of range-compressed data
    for (int k = kstart_min; k < kstop_max; k += batch) {

        // actual size of current batch
        const int curr_batch = std::min(batch, kstop_max - k);

        // copy batch of range-compressed data to device memory
        checkCudaErrors(
                cudaMemcpy(rc.data().get(), &in[k * width],
                           curr_batch * width * sizeof(std::complex<float>),
                           cudaMemcpyHostToDevice));

        // integrate pulses
        const unsigned block = 256;
        const unsigned grid = (out_grid_size + block - 1) / block;

        using KV = typename Kernel::view_type;

        sumCoherentBatch<KV><<<grid, block>>>(
                img.data().get(), rc.data().get(), pos.data().get(),
                vel.data().get(), sampling_window, x.data().get(),
                tau_atm.data().get(), kstart.data().get(), kstop.data().get(),
                out_grid_size, fc, kernel, k, k + curr_batch);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    // copy output back to the host
    checkCudaErrors(cudaMemcpy(out, img.data().get(),
                               out_grid_size * sizeof(std::complex<float>),
                               cudaMemcpyDeviceToHost));

    return errc[0];
}

ErrorCode backproject(std::complex<float>* out,
                      const HostRadarGeometry& out_geometry,
                      const std::complex<float>* in,
                      const HostRadarGeometry& in_geometry,
                      const HostDEMInterpolator& dem, double fc, double ds,
                      const Kernel<float>& kernel,
                      DryTroposphereModel dry_tropo_model,
                      const Rdr2GeoBracketParams& rdr2geo_params,
                      const Geo2RdrBracketParams& geo2rdr_params, int batch,
                      float* height)
{
    // copy inputs to device
    const DeviceRadarGeometry d_out_geometry(out_geometry);
    const DeviceRadarGeometry d_in_geometry(in_geometry);
    DeviceDEMInterpolator d_dem(dem);
    ErrorCode ec;

    if (typeid(kernel) == typeid(HostBartlettKernel<float>)) {
        const DeviceBartlettKernel<float> d_kernel(
                dynamic_cast<const HostBartlettKernel<float>&>(kernel));
        ec = backproject(out, d_out_geometry, in, d_in_geometry, d_dem, fc, ds,
                         d_kernel, dry_tropo_model, rdr2geo_params,
                         geo2rdr_params, batch, height);
    }
    else if (typeid(kernel) == typeid(HostLinearKernel<float>)) {
        const DeviceLinearKernel<float> d_kernel(
                dynamic_cast<const HostLinearKernel<float>&>(kernel));
        ec = backproject(out, d_out_geometry, in, d_in_geometry, d_dem, fc, ds,
                         d_kernel, dry_tropo_model, rdr2geo_params,
                         geo2rdr_params, batch, height);
    }
    else if (typeid(kernel) == typeid(HostKnabKernel<float>)) {
        const DeviceKnabKernel<float> d_kernel(
                dynamic_cast<const HostKnabKernel<float>&>(kernel));
        ec = backproject(out, d_out_geometry, in, d_in_geometry, d_dem, fc, ds,
                         d_kernel, dry_tropo_model, rdr2geo_params,
                         geo2rdr_params, batch, height);
    }
    else if (typeid(kernel) == typeid(HostTabulatedKernel<float>)) {
        const DeviceTabulatedKernel<float> d_kernel(
                dynamic_cast<const HostTabulatedKernel<float>&>(kernel));
        ec = backproject(out, d_out_geometry, in, d_in_geometry, d_dem, fc, ds,
                         d_kernel, dry_tropo_model, rdr2geo_params,
                         geo2rdr_params, batch, height);
    }
    else if (typeid(kernel) == typeid(HostChebyKernel<float>)) {
        const DeviceChebyKernel<float> d_kernel(
                dynamic_cast<const HostChebyKernel<float>&>(kernel));
        ec = backproject(out, d_out_geometry, in, d_in_geometry, d_dem, fc, ds,
                         d_kernel, dry_tropo_model, rdr2geo_params,
                         geo2rdr_params, batch, height);
    }
    else {
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), "not implemented");
    }
    return ec;
}


// FBP junk

/**
 * \internal
 * Transform a 2D radar grid from polar coordinates (cos_squint, range) to
 * ECEF XYZ coordinates.
 *
 * The global error code is set if any thread encounters an error.
 *
 * \param[out] xyz_out      ECEF XYZ of each target (m)
 * \param[in]  grid         Polar grid
 * \param[in]  dem          DEM sampling interface
 * \param[in]  ellipsoid    Reference ellipsoid
 * \param[in]  side         Radar look side
 * \param[in]  params       Root-finding algorithm parameters
 * \param[out] errc         Error flag
 */
__global__ void runPolar2Geo(Vec3* xyz_out, const PolarGrid grid,
                           DeviceDEMInterpolator dem, const Ellipsoid ellipsoid,
                           const LookSide side,
                           const Rdr2GeoBracketParams params,
                           ErrorCode* errc)
{
    using isce3::geometry::detail::polar2geo_bracket;

    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    const auto lines = static_cast<size_t>(grid.length());
    const auto samples = static_cast<size_t>(grid.width());
    if (tid >= lines * samples) {
        return;
    }

    // convert flat index to 2D array indices
    const auto j = static_cast<int>(tid / samples);
    const auto i = static_cast<int>(tid % samples);

    const double r = grid.range[i];
    const double q = grid.sin_squint[j];
    const double c = sqrt(1.0 - q * q);

    Vec3 xyz;
    double look_angle;

    const auto status = polar2geo_bracket(&xyz, &look_angle,
                        grid.origin, grid.axis, r, q, c, dem, ellipsoid,
                        side, params);

    // check convergence
    if (status == isce3::error::ErrorCode::Success) {
        xyz_out[tid] = xyz;
    } else {
        // set output to NaN
        constexpr static auto nan = std::numeric_limits<double>::quiet_NaN();
        xyz_out[tid] = {nan, nan, nan};

        // set global error flag
        *errc = ErrorCode::FailedToConverge;
    }
}


template <class Clock = std::chrono::high_resolution_clock>
class TimingReporter {
public:
    TimingReporter(const std::string& channel_id) :
        prev_time_{Clock::now()}, log_{channel_id} {}

    void report(const std::string& step) {
        using namespace std::chrono;
        auto cur_time = Clock::now();
        auto duration = cur_time - prev_time_;
        auto msec = duration_cast<milliseconds>(duration).count();
        log_ << step << " took " << msec << " ms" << pyre::journal::endl;
        prev_time_ = cur_time;
    }
private:
    std::chrono::time_point<Clock> prev_time_;
    pyre::journal::info_t log_;
};

// macro to enable logging of timing info
#ifdef ISCE3_ENABLE_FBP_TIMING
#define ISCE3_FBP_TIMING(x) x
#else
#define ISCE3_FBP_TIMING(x)  // no-op
#endif

template<class Kernel>
std::tuple<ErrorCode, std::unique_ptr<std::complex<float>[]>, std::unique_ptr<float[]>>
backprojectToPolarGrid(
        const std::complex<float>* in,
        const Linspace<double>& in_slant_range,
        const std::vector<Vec3>& pos,
        const std::vector<Vec3>& vel,
        const PolarGrid& out_grid,
        DeviceDEMInterpolator& dem, double fc,
        const Kernel& kernel, DryTroposphereModel dry_tropo_model,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params)
{
    const auto nt = static_cast<int>(pos.size());
    if (nt < 2) {
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "require at least two pulses in FBP stage");
    }
    if (vel.size() != nt) {
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "require same number of position and velocity vectors");
    }

    static constexpr double c = isce3::core::speed_of_light;

    // check that dry_tropo_model is supported internally
    if (not(dry_tropo_model == DryTroposphereModel::NoDelay or
            dry_tropo_model == DryTroposphereModel::TSX)) {

        std::string errmsg = "unexpected dry troposphere model";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    ISCE3_FBP_TIMING(
        auto timing = TimingReporter("isce3.cuda.focus.backprojectToPolarGrid");
    )

    const auto npix = static_cast<size_t>(out_grid.length()) * out_grid.width();
    auto height = std::make_unique<float[]>(npix);
    auto out = std::make_unique<std::complex<float>[]>(npix);

    // range sampling window
    double swst = 2. * in_slant_range.first() / c;
    double dtau = 2. * in_slant_range.spacing() / c;
    int nr = in_slant_range.size();
    Linspace<double> sampling_window(swst, dtau, nr);

    // reference ellipsoid
    int epsg = dem.epsgCode();
    const Ellipsoid ellipsoid = makeProjection(epsg)->ellipsoid();

    // init device variable to return error codes from device code
    thrust::device_vector<ErrorCode> errc(1, ErrorCode::Success);

    thrust::device_vector<Vec3> x(npix);

    {
        const unsigned block = 256;
        const unsigned grid = (npix + block - 1) / block;

        runPolar2Geo<<<grid, block>>>(x.data().get(), out_grid,
                                    dem, ellipsoid,
                                    out_grid.look_side, r2g_params,
                                    errc.data().get());

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }
    ISCE3_FBP_TIMING(timing.report("polar2geo");)

    // transform each target position from ECEF to LLH coordinates
    // NOTE only really needed if dumping height layer or doing TSX atmosphere
    // correction, but just compute it unconditionally.
    thrust::device_vector<Vec3> llh(npix);

    {
        const unsigned block = 256;
        const unsigned grid = (npix + block - 1) / block;

        ecef2llh<<<grid, block>>>(llh.data().get(), x.data().get(),
                                  npix, ellipsoid);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    if (height != nullptr) {
        thrust::device_vector<float> d_height(npix);
        thrust::transform(llh.begin(), llh.end(), d_height.begin(),
                [] __device__ (const Vec3& x) { return (float)x[2]; });
        checkCudaErrors(cudaMemcpy(height.get(), d_height.data().get(),
                npix * sizeof(float), cudaMemcpyDeviceToHost));
    }
    ISCE3_FBP_TIMING(timing.report("ecef2llh");)

    // estimate dry troposphere delay
    thrust::device_vector<double> tau_atm(npix);

    if (dry_tropo_model == DryTroposphereModel::NoDelay) {
        checkCudaErrors(cudaMemset(tau_atm.data().get(), 0,
                                   npix * sizeof(double)));
    } else if (dry_tropo_model == DryTroposphereModel::TSX) {
        const unsigned block = 256;
        const unsigned grid = (npix + block - 1) / block;

        // TODO new interface for constant aperture center
        thrust::device_vector<Vec3> p(npix);
        thrust::fill(p.begin(), p.end(), out_grid.origin);

        estimateDryTropoDelayTSX<<<grid, block>>>(
                tau_atm.data().get(), p.data().get(), llh.data().get(),
                npix, ellipsoid);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    } else {
        std::string errmsg = "unexpected dry troposphere model";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }
    ISCE3_FBP_TIMING(timing.report("dry troposphere");)

    // Assume we can fit all pulses for a subimage in device memory.
    const auto npix_in = static_cast<size_t>(in_slant_range.size()) * nt;
    thrust::device_vector<thrust::complex<float>> rc(npix_in);
    checkCudaErrors(cudaMemcpy(rc.data().get(), in, npix_in * sizeof(*in),
        cudaMemcpyHostToDevice));

    thrust::device_vector<thrust::complex<float>> img(npix, 0.0);
	{
        // integrate pulses
        const unsigned block = 256;
        const unsigned grid = (npix + block - 1) / block;

        using KV = typename Kernel::view_type;

        thrust::device_vector<Vec3> d_pos(pos);
        thrust::device_vector<Vec3> d_vel(vel);

        // TODO interface with scalar kstart & kstop
        thrust::device_vector<int> kstart(npix, 0);
        thrust::device_vector<int> kstop(npix, nt);

        sumCoherentBatch<KV><<<grid, block>>>(
                img.data().get(), rc.data().get(), d_pos.data().get(),
                d_vel.data().get(), sampling_window, x.data().get(),
                tau_atm.data().get(), kstart.data().get(), kstop.data().get(),
                npix, fc, kernel, 0, nt);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }
    ISCE3_FBP_TIMING(timing.report("sum coherent");)

    // copy output back to the host
    checkCudaErrors(cudaMemcpy(out.get(), img.data().get(),
                               npix * sizeof(std::complex<float>),
                               cudaMemcpyDeviceToHost));


    // baseband
    const double kw = 4 * M_PI / (c / fc);
    #pragma omp parallel for
    for (int i = 0; i < out_grid.range.size(); ++i) {
        const double phi = -kw * out_grid.range[i];
        const auto phasor = std::complex<float>(std::cos(phi), std::sin(phi));
        for (int j = 0; j < out_grid.sin_squint.size(); ++j) {
            out[j * out_grid.width() + i] *= phasor;
        }
    }
    ISCE3_FBP_TIMING(timing.report("baseband");)

    return std::make_tuple(errc[0], std::move(out), std::move(height));
}


std::tuple<
    ErrorCode,
    std::unique_ptr<std::complex<float>[]>,
    std::unique_ptr<float[]>>
backprojectToPolarGrid(
        const std::complex<float>* in, const Linspace<double>& in_slant_range,
        const std::vector<Vec3>& pos, const std::vector<Vec3>& vel,
        const PolarGrid& out_grid,
        const HostDEMInterpolator& dem, double fc,
        const Kernel<float>& kernel, DryTroposphereModel dry_tropo_model,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params)
{
    DeviceDEMInterpolator d_dem(dem);

    if (typeid(kernel) == typeid(HostBartlettKernel<float>)) {
        const DeviceBartlettKernel<float> d_kernel(
                dynamic_cast<const HostBartlettKernel<float>&>(kernel));
        return backprojectToPolarGrid(in, in_slant_range, pos, vel,
            out_grid, d_dem, fc, d_kernel, dry_tropo_model, r2g_params);
    }
    else if (typeid(kernel) == typeid(HostLinearKernel<float>)) {
        const DeviceLinearKernel<float> d_kernel(
                dynamic_cast<const HostLinearKernel<float>&>(kernel));
        return backprojectToPolarGrid(in, in_slant_range, pos, vel,
            out_grid, d_dem, fc, d_kernel, dry_tropo_model, r2g_params);
    }
    else if (typeid(kernel) == typeid(HostKnabKernel<float>)) {
        const DeviceKnabKernel<float> d_kernel(
                dynamic_cast<const HostKnabKernel<float>&>(kernel));
        return backprojectToPolarGrid(in, in_slant_range, pos, vel,
            out_grid, d_dem, fc, d_kernel, dry_tropo_model, r2g_params);
    }
    else if (typeid(kernel) == typeid(HostTabulatedKernel<float>)) {
        const DeviceTabulatedKernel<float> d_kernel(
                dynamic_cast<const HostTabulatedKernel<float>&>(kernel));
        return backprojectToPolarGrid(in, in_slant_range, pos, vel,
            out_grid, d_dem, fc, d_kernel, dry_tropo_model, r2g_params);
    }
    else if (typeid(kernel) == typeid(HostChebyKernel<float>)) {
        const DeviceChebyKernel<float> d_kernel(
                dynamic_cast<const HostChebyKernel<float>&>(kernel));
        return backprojectToPolarGrid(in, in_slant_range, pos, vel,
            out_grid, d_dem, fc, d_kernel, dry_tropo_model, r2g_params);
    }
    throw isce3::except::RuntimeError(ISCE_SRCINFO(), "not implemented");
}

__global__ void
interpPolar(thrust::complex<float>* geo_image, const Vec3* geo_points,
    size_t n, const PolarGrid grid,
    const NFFT2dResultView<float> nfft, const double kw,
    std::optional<const bool*> mask = std::nullopt,
    std::optional<const double*> extra_range_delays = std::nullopt)
{
    // thread index (1d grid of 1d blocks)
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // bounds check
    if (i >= n) {
        return;
    }
    // mask check
    if (mask.has_value() and not mask.value()[i]) {
        return;
    }

    // compute target location in polar grid
    // TODO not sure why I get a linker error when I try using the API (with CUDA_HOSTDEV added)
    // double sin_squint, range;
    // isce3::geometry::geo2polar(&sin_squint, &range, geo_points[i], grid.origin, grid.axis);
    const Vec3 lookvec = geo_points[i] - grid.origin;
    double range = lookvec.norm();
    const double sin_squint = lookvec.dot(grid.axis) / range;

    if (extra_range_delays.has_value()) {
        range += extra_range_delays.value()[i];
    }

    // convert to image index
    const double ix = (range - grid.range.first()) / grid.range.spacing(),
        iy = (sin_squint - grid.sin_squint.first()) / grid.sin_squint.spacing();

    // interpolate baseband data
    const auto z = nfft.interp({iy, ix}, /* periodic */ false);

    // compensate phase and sum contribution
    double sin_phi, cos_phi;
    ::sincos(kw * range, &sin_phi, &cos_phi);
    geo_image[i] += z * thrust::complex<float>(cos_phi, sin_phi);
}

ErrorCode
projectPolarToGeo(
        std::complex<float>* geo_image,
        const Vec3* geo_points,
        const size_t n,
        const PolarGrid& grid,
        const std::complex<float>* polar_image,
        const double wavelength,
        const isce3::signal::NFFT2dParams& params)
{
    using isce3::fft::nextFastPower;
    using dims_t = isce3::cuda::signal::NFFT2d<float>::dims_t;
    using std::lround;

    const dims_t m = {params.rows.m, params.cols.m};

    // NFFTKernel width = 2*m+1; interp2d uses fixed-size stack arrays of
    // MAX_WIDTH=16, so reject m >= 8 here at the API entry point instead of
    // corrupting memory in the device kernel.
    constexpr int MAX_NFFT_M = 7;
    if (m[0] > MAX_NFFT_M or m[1] > MAX_NFFT_M) {
        return ErrorCode::InvalidKernelSize;
    }

    const dims_t dims_in = {grid.length(), grid.width()};
    const dims_t dims_out = {
        nextFastPower(static_cast<int32_t>(lround(params.rows.s * dims_in[0]))),
        nextFastPower(static_cast<int32_t>(lround(params.cols.s * dims_in[1])))
    };

    auto nfft = isce3::cuda::signal::NFFT2d<float>(m, dims_in, dims_out);
    const size_t nin = static_cast<size_t>(grid.length()) * grid.width();
    std::vector<std::complex<float>> spectrum(nin);
    // Okay to discard const because fft is planned with FFTW_EXECUTE which
    // doesn't modify input.
    ISCE3_FBP_TIMING(
        auto timing = TimingReporter("isce3.cuda.focus.backprojectToPolarGrid");)
    // TODO do this FFT on GPU
    isce3::fft::fft2d(spectrum.data(),
        const_cast<std::complex<float>*>(polar_image),
        {dims_in[0], dims_in[1]});
    ISCE3_FBP_TIMING(timing.report("FFT");)

    // zero-pad and filter
    auto result = nfft.transform_host(dims_in, /* strides = */ {dims_in[1], 1},
        spectrum.data());
    auto nfft_view = NFFT2dResultView<float>(result);
    ISCE3_FBP_TIMING(timing.report("IFFT");)

    const double kw = 4 * M_PI / wavelength;

    // TODO add interface that just accepts device pointers as argument?
    // NOTE You can construct device_vector using iterators, but that's super
    // slow, hence the manual copy.
    thrust::device_vector<thrust::complex<float>> d_geo_image(n);
    thrust::device_vector<Vec3> d_geo_points(n);
    checkCudaErrors(cudaMemcpy(d_geo_image.data().get(), geo_image,
        n * sizeof(*geo_image), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_geo_points.data().get(), geo_points,
        n * sizeof(*geo_points), cudaMemcpyHostToDevice));
    ISCE3_FBP_TIMING(timing.report("copy to device");)

    {
        const unsigned block = 256;
        const unsigned cugrid = (n + block - 1) / block;

        interpPolar<<<cugrid, block>>>(d_geo_image.data().get(),
            d_geo_points.data().get(), n, grid, nfft_view, kw);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }
    ISCE3_FBP_TIMING(timing.report("interp");)

    checkCudaErrors(cudaMemcpy(geo_image, d_geo_image.data().get(),
        n * sizeof(*geo_image), cudaMemcpyDeviceToHost));
    ISCE3_FBP_TIMING(timing.report("copy to host");)

    return ErrorCode::Success;
}


isce3::error::ErrorCode
accumulatePolarImagesToRadarGrid(std::complex<float>* out,
        const isce3::container::RadarGeometry& out_geometry,
        const isce3::core::Orbit& in_orbit,
        const isce3::core::LUT2d<double>& in_doppler,
        const std::vector<isce3::focus::PolarGrid>& grids,
        const std::vector<const NFFT2dResult<float>*>& image_interpolators,
        const isce3::geometry::DEMInterpolator& dem, double fc, double ds,
        DryTroposphereModel dry_tropo_model,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
        const isce3::geometry::detail::Geo2RdrBracketParams& g2r_params,
        float* height)
{
    using namespace isce3::core;
    using namespace isce3::cuda::geometry;
    using isce3::focus::dryTropoDelayTSX;
    using isce3::error::ErrorCode;
    using isce3::focus::PolarGrid;
    using DeviceDEMInterpolator = isce3::cuda::geometry::gpuDEMInterpolator;
    using DeviceRadarGeometry = isce3::cuda::container::RadarGeometry;

    static constexpr double c = isce3::core::speed_of_light;
    static constexpr auto nan = std::numeric_limits<float>::quiet_NaN();

    // will search sorted intervals to figure out active sub images per target
    auto starts = std::vector<double>(grids.size());
    std::transform(grids.begin(), grids.end(), starts.begin(),
        [](const PolarGrid& grid) { return grid.aztime_start; });
    auto ends = std::vector<double>(grids.size());
    std::transform(grids.begin(), grids.end(), ends.begin(),
        [](const PolarGrid& grid) { return grid.aztime_end; });

    // get input & output radar grid azimuth time & slant range
    Linspace<double> out_azimuth_time = out_geometry.sensingTime();
    Linspace<double> out_slant_range = out_geometry.slantRange();

    // reference ellipsoid
    int epsg = dem.epsgCode();
    Ellipsoid ellipsoid = makeProjection(epsg)->ellipsoid();

    // carrier wavelength
    const double wvl = c / fc;
    const double kw = 4 * M_PI / wvl;

    // copy inputs to device
    const DeviceRadarGeometry d_out_geometry(out_geometry);
    DeviceDEMInterpolator d_dem(dem);

    const size_t nout = out_geometry.gridLength() * out_geometry.gridWidth();

    thrust::device_vector<Vec3> d_x(nout);
    thrust::device_vector<ErrorCode> errc(1, ErrorCode::Success);
    {
        const unsigned block = 256;
        const unsigned grid = (nout + block - 1) / block;

        runRdr2Geo<<<grid, block>>>(d_x.data().get(), out_azimuth_time,
                out_slant_range, d_out_geometry.doppler(),
                d_out_geometry.orbit(), d_dem, ellipsoid, wvl,
                d_out_geometry.lookSide(), r2g_params, errc.data().get());

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    // transform each target position from ECEF to LLH coordinates
    // NOTE only really needed if dumping height layer or doing TSX atmosphere
    // correction, but just compute it unconditionally.
    thrust::device_vector<Vec3> d_llh(nout);

    {
        const unsigned block = 256;
        const unsigned grid = (nout + block - 1) / block;

        ecef2llh<<<grid, block>>>(d_llh.data().get(), d_x.data().get(),
                                  nout, ellipsoid);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    if (height != nullptr) {
        thrust::device_vector<float> d_height(nout);
        thrust::transform(d_llh.begin(), d_llh.end(), d_height.begin(),
                [] __device__ (const Vec3& x) { return (float)x[2]; });
        checkCudaErrors(cudaMemcpy(height, d_height.data().get(),
                nout * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Running geo2rdr to get integration bounds seems like overkill.
    // TODO Maybe mask on Doppler instead?
    thrust::device_vector<double> d_t(nout);
    thrust::device_vector<double> d_r(nout);
    auto d_in_orbit = isce3::cuda::core::Orbit(in_orbit);
    auto d_in_orbit_view = isce3::cuda::core::OrbitView(d_in_orbit);
    auto d_in_doppler = isce3::cuda::core::gpuLUT2d<double>(in_doppler);

    {
        const unsigned block = 256;
        const unsigned grid = (nout + block - 1) / block;

        runGeo2Rdr<<<grid, block>>>(
                d_t.data().get(), d_r.data().get(), d_x.data().get(), nout,
                d_in_orbit_view, d_in_doppler, wvl,
                d_out_geometry.lookSide(), g2r_params, errc.data().get());

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    // get platform position & velocity at center of CPI for each target
    thrust::device_vector<Vec3> d_p(nout);
    thrust::device_vector<Vec3> d_v(nout);

    {
        const unsigned block = 256;
        const unsigned grid = (nout + block - 1) / block;

        interpolateOrbit<<<grid, block>>>(d_p.data().get(), d_v.data().get(),
                                          d_in_orbit_view, d_t.data().get(),
                                          nout, errc.data().get());

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    // Calculate dry troposphere delay.  To re-use code and memory, first we'll
    // calculate in time units and then convert to spatial units.
    thrust::device_vector<double> dr_atm(nout);

    if (dry_tropo_model == DryTroposphereModel::NoDelay) {
        checkCudaErrors(cudaMemset(dr_atm.data().get(), 0,
                                   nout * sizeof(double)));
    } else if (dry_tropo_model == DryTroposphereModel::TSX) {
        const unsigned block = 256;
        const unsigned grid = (nout + block - 1) / block;

        estimateDryTropoDelayTSX<<<grid, block>>>(
                dr_atm.data().get() /* time units */, d_p.data().get(),
                d_llh.data().get(), nout, ellipsoid);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    } else {
        std::string errmsg = "unexpected dry troposphere model";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    // two-way time -> one-way range
    constexpr double halfspeed = isce3::core::speed_of_light / 2.0;
    thrust::transform(dr_atm.begin(), dr_atm.end(), dr_atm.begin(),
        [halfspeed] __host__ __device__ (double dt) { return halfspeed * dt; });

    d_llh.clear();  d_llh.shrink_to_fit();

    // get coherent integration bounds (pulse indices) for each target
    thrust::device_vector<double> d_tstart(nout);
    thrust::device_vector<double> d_tstop(nout);

    {
        const unsigned block = 256;
        const unsigned grid = (nout + block - 1) / block;

        getCPITimeBounds<<<grid, block>>>(d_tstart.data().get(),
                d_tstop.data().get(), d_t.data().get(), d_r.data().get(),
                d_x.data().get(), d_p.data().get(), d_v.data().get(), nout,
                wvl, ds);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    d_p.clear();  d_p.shrink_to_fit();
    d_v.clear();  d_v.shrink_to_fit();
    d_t.clear();  d_t.shrink_to_fit();
    d_r.clear();  d_r.shrink_to_fit();

    // NOTE Thrust does not use the bit-packing strategy as STL.
    thrust::device_vector<bool> d_mask(nout);

    // TODO reduce tstart & tend
    // TODO check this O(log(n)) algorithm
    //const auto kstart = std::distance(ends.begin(),
    //    std::lower_bound(ends.begin(), ends.end(), tstart));
    //const auto kstop = std::distance(starts.begin(),
    //    std::upper_bound(starts.start(), starts.end(), tstart + cpi));
    const auto num_images = image_interpolators.size();
    const decltype(num_images) kstart = 0, kstop = num_images;

    // Copy image to device since we accumulate (don't init to zero).
    // thrust::device_vector<thrust::complex<float>> d_out(out, out + nout);
    thrust::device_vector<thrust::complex<float>> d_out(nout);
    checkCudaErrors(cudaMemcpy(d_out.data().get(), out, nout * sizeof(*out),
        cudaMemcpyHostToDevice));

    // Advance forward iterator as needed.
    auto image_iter = image_interpolators.begin();
    for (int k = 0; k < kstart; ++k) ++image_iter;

    for (auto k = kstart; k < kstop; ++k, ++image_iter) {
        const auto& image_grid = grids[k];
        const auto d_nfft = *image_iter;

        {
            const unsigned block = 256;
            const unsigned cuda_grid = (nout + block - 1) / block;

            makeSubApertureMask<<<cuda_grid, block>>>(image_grid.aztime_start,
                image_grid.aztime_end, nout, d_tstart.data().get(),
                d_tstop.data().get(), d_mask.data().get());

            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
        }

        const auto d_nfft_view = NFFT2dResultView(*d_nfft);

        {
            const unsigned block = 256;
            const unsigned cuda_grid = (nout + block - 1) / block;

            interpPolar<<<cuda_grid, block>>>(d_out.data().get(),
                    d_x.data().get(), nout, image_grid, d_nfft_view, kw,
                    d_mask.data().get(), dr_atm.data().get());

            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
        }
    }

    // Copy result back to host.
    checkCudaErrors(cudaMemcpy(out, d_out.data().get(), nout * sizeof(*out),
        cudaMemcpyDeviceToHost));

    return errc[0];
}


template <class NFFT2dSequence>
void mergePolarImages(
    const std::vector<isce3::focus::PolarGrid>& grids,
    const std::vector<const NFFT2dResult<float>*>& image_interpolators,
    const isce3::focus::PolarGrid& output_grid,
    Eigen::Ref<isce3::core::EArray2D<std::complex<float>>> output_image,
    const double fc,
    const isce3::geometry::DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
    int az_block_size)
{
    // check that output grid dimensions match buffer size
    const auto m = output_grid.length(), n = output_grid.width();
    if ((m != output_image.rows()) or (n != output_image.cols())) {
        std::string msg = "Dimensions of image grid (" + std::to_string(m)
            + ", " + std::to_string(n) + ") do not match dimensions of image "
            "buffer (" + std::to_string(output_image.rows()) + ", "
            + std::to_string(output_image.cols()) + ")";
        throw isce3::except::LengthError(ISCE_SRCINFO(), msg);
    }

    // check that we have a grid for each input image
    const auto num_images = image_interpolators.size();
    if (grids.size() != num_images) {
        std::string msg = "Size mismatch: got " + std::to_string(num_images) +
            " sub images but " + std::to_string(grids.size()) + " grids";
        throw isce3::except::LengthError(ISCE_SRCINFO(), msg);
    }

    // check look directions for consistency
    const auto look_side = output_grid.look_side;
    for (const auto& grid : grids) {
        if (grid.look_side != look_side) {
            std::string msg = "Output grid look direction does not match "
                "input grid look direction";
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(), msg);
        }
    }

    // Check block size and allocate scratch space.
    if (az_block_size <= 0) {
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "azimuth block size must be positive");
    }
    az_block_size = std::min(az_block_size, output_grid.sin_squint.size());

    // auto block_positions = isce3::core::EArray2D<Vec3>();
    const auto npix = static_cast<size_t>(az_block_size) * output_grid.width();
    auto block_positions = thrust::device_vector<isce3::core::Vec3>(npix);
    auto block_image = thrust::device_vector<thrust::complex<float>>(npix);

    // reference ellipsoid
    isce3::core::Ellipsoid ellipsoid = isce3::core::makeProjection(dem.epsgCode())->ellipsoid();
    isce3::cuda::geometry::gpuDEMInterpolator d_dem(dem);

    // wavenumber
    const double kw = 4 * M_PI * fc / isce3::core::speed_of_light;

    // Baseband.  Note that we could do this at the same time as the
    // reprojection but it'd require a fair bit of copy/paste.
    thrust::host_vector<thrust::complex<float>> h_phasors(n);
    #pragma omp parallel for
    for (auto j = decltype(n){0}; j < n; ++j) {
        const double arg = -kw * output_grid.range[j];
        h_phasors[j] = std::complex<float>(std::cos(arg), std::sin(arg));
    }
    const auto d_phasors = thrust::device_vector<thrust::complex<float>>(h_phasors);

    using isce3::error::ErrorCode;
    thrust::device_vector<ErrorCode> errc(1, ErrorCode::Success);

    // loop over output blocks
    auto n_blocks = (m + az_block_size - 1) / az_block_size;
    for (auto i_block = decltype(n_blocks){0}; i_block < n_blocks; ++i_block) {
        const auto i_row0 = i_block * az_block_size;
        const auto i_row1 = std::min(i_row0 + az_block_size, m);
        const auto block_npix = static_cast<size_t>(n) * (i_row1 - i_row0);

        // Zero out block_image since interpPolar accumulates
        checkCudaErrors(cudaMemset(block_image.data().get(), 0,
                                   block_npix * sizeof(thrust::complex<float>)));

        const auto output_grid_subset = output_grid.offsetAndResize(i_row0, 0,
            i_row1 - i_row0, output_grid.width());

        // Compute output pixel 3D locations
        {
            const unsigned cu_block = 256;
            const unsigned cu_grid = (block_npix + cu_block - 1) / cu_block;

            runPolar2Geo<<<cu_grid, cu_block>>>(block_positions.data().get(),
                    output_grid_subset, d_dem, ellipsoid, output_grid.look_side,
                    r2g_params, errc.data().get());

            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
        }
        const auto ec = errc[0];
        if (ec != ErrorCode::Success) {
            throw isce3::except::DomainError(ISCE_SRCINFO(),
                "polar2geo failed with ErrorCode " +
                isce3::error::getErrorString(ec));
        }

        // loop over input images
        auto image_it = image_interpolators.begin();
        for (int i_img = 0; i_img < num_images; ++i_img, ++image_it) {
            const auto& input_grid = grids[i_img];
            const auto d_nfft = *image_it;
            const auto d_nfft_view = NFFT2dResultView<float>(*d_nfft);
            {
                const unsigned cu_block = 256;
                const unsigned cu_grid = (block_npix + cu_block - 1) / cu_block;

                interpPolar<<<cu_grid, cu_block>>>(block_image.data().get(),
                    block_positions.data().get(), block_npix, input_grid,
                    d_nfft_view, kw);

                checkCudaErrors(cudaPeekAtLastError());
                checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
            }
        } // images

        // baseband
        {
            const unsigned cu_block = 256;
            const unsigned cu_grid = (block_npix + cu_block - 1) / cu_block;

            broadcastMultiply<<<cu_grid, cu_block>>>(d_phasors.data().get(),
                    n, block_image.data().get(), block_npix);

            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
        }

        checkCudaErrors(cudaMemcpy(output_image.row(i_row0).data(),
            block_image.data().get(), block_npix * sizeof(std::complex<float>),
            cudaMemcpyDeviceToHost));
    } // blocks
}

}}} // namespace isce3::cuda::focus
