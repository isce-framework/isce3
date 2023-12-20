#include "Backproject.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
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
#include <isce3/cuda/geometry/gpuDEMInterpolator.h>
#include <isce3/cuda/geometry/gpuGeometry.h>
#include <isce3/focus/BistaticDelay.h>

using namespace isce3::core;
using namespace isce3::cuda::geometry;

using isce3::cuda::core::interp1d;
using isce3::error::ErrorCode;
using isce3::focus::bistaticDelay;
using isce3::focus::dryTropoDelayTSX;

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

namespace {

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

} // namespace

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

}}} // namespace isce3::cuda::focus
