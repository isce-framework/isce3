#include "Backproject.h"

#include <cmath>
#include <isce3/container/RadarGeometry.h>
#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Interp1d.h>
#include <isce3/core/Interp2d.h>
#include <isce3/core/Kernels.h>
#include <isce3/core/Projections.h>
#include <isce3/except/Error.h>
#include <isce3/fft/FFT.h>
#include <isce3/fft/FFTUtil.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/geometry.h>
#include <isce3/geometry/rdr2geo_roots.h>
#include <isce3/geometry/geo2rdr_roots.h>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "BistaticDelay.h"

using namespace isce3::core;
using namespace isce3::geometry;
using isce3::error::ErrorCode;

using isce3::container::RadarGeometry;
using isce3::signal::NFFT2d;
using isce3::signal::NFFT2dParams;
using isce3::fft::planfft2d;
using isce3::fft::nextFastPower;

namespace isce3 {
namespace focus {

inline std::complex<float> sumCoherent(const std::complex<float>* data,
                                       const Linspace<double>& sampling_window,
                                       const std::vector<Vec3>& pos,
                                       const std::vector<Vec3>& vel,
                                       const Vec3& x,
                                       double fc,
                                       double tau_atm,
                                       const Kernel<float>& kernel,
                                       int kstart, int kstop)
{
    // loop over pulses within integration window
    std::complex<double> sum(0., 0.);
    for (int k = kstart; k < kstop; ++k) {

        // compute round-trip delay to target
        double tau = tau_atm + bistaticDelay(pos[k], vel[k], x);

        // interpolate range-compressed data
        auto data_line = &data[size_t(k) * sampling_window.size()];
        double u = (tau - sampling_window.first()) / sampling_window.spacing();
        std::complex<double> s =
                interp1d(kernel, data_line, sampling_window.size(), 1, u);

        // apply phase migration compensation
        double phi = 2. * M_PI * fc * tau;
        s *= std::complex<double>(std::cos(phi), std::sin(phi));

        // worst-case numerical error increases linearly, accumulate using
        // double precision to mitigate errors
        sum += s;
    }

    return std::complex<float>(sum);
}

ErrorCode
backproject(std::complex<float>* out, const RadarGeometry& out_geometry,
        const std::complex<float>* in, const RadarGeometry& in_geometry,
        const DEMInterpolator& dem, double fc, double ds,
        const Kernel<float>& kernel, DryTroposphereModel dry_tropo_model,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
        const isce3::geometry::detail::Geo2RdrBracketParams& g2r_params,
        float* height)
{
    static constexpr double c = isce3::core::speed_of_light;
    static constexpr auto nan = std::numeric_limits<float>::quiet_NaN();

    // check that dry_tropo_model is supported internally
    if (not(dry_tropo_model == DryTroposphereModel::NoDelay or
            dry_tropo_model == DryTroposphereModel::TSX)) {

        std::string errmsg = "unexpected dry troposphere model";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    // XXX not very nice to throw here instead of simply adjusting the epoch
    // XXX but doing so at this point would require making a copy of the input
    // XXX radar grid, orbit, and Doppler - so this is just a stopgap for now
    if (out_geometry.referenceEpoch() != in_geometry.referenceEpoch()) {
        std::string errmsg = "input reference epoch must match output "
                             "reference epoch";
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), errmsg);
    }

    // get input & output radar grid azimuth time & slant range
    Linspace<double> in_azimuth_time = in_geometry.sensingTime();
    Linspace<double> in_slant_range = in_geometry.slantRange();
    Linspace<double> out_azimuth_time = out_geometry.sensingTime();
    Linspace<double> out_slant_range = out_geometry.slantRange();

    // interpolate platform position & velocity at each pulse
    std::vector<Vec3> pos(in_azimuth_time.size());
    std::vector<Vec3> vel(in_azimuth_time.size());
    for (int i = 0; i < in_azimuth_time.size(); ++i) {
        double t = in_azimuth_time[i];
        in_geometry.orbit().interpolate(&pos[i], &vel[i], t);
    }

    // range sampling window
    double swst = 2. * in_slant_range.first() / c;
    double dtau = 2. * in_slant_range.spacing() / c;
    int nr = in_slant_range.size();
    Linspace<double> sampling_window(swst, dtau, nr);

    // reference ellipsoid
    int epsg = dem.epsgCode();
    Ellipsoid ellipsoid = makeProjection(epsg)->ellipsoid();

    // carrier wavelength
    double wvl = c / fc;

    // loop over targets in output grid
    bool all_converged = true;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < out_azimuth_time.size(); ++j) {
        for (int i = 0; i < out_slant_range.size(); ++i) {

            // Run rdr2geo using orbit and Doppler associated with output grid
            // to get target position.  Only need LLH if dumping height or
            // using TSX atmosphere model, but just compute it unconditionally.
            Vec3 x, llh;
            {
                double t = out_azimuth_time[j];
                double r = out_slant_range[i];
                double fD = out_geometry.doppler().eval(t, r);

                const int converged = rdr2geo_bracket(t, r, fD,
                        out_geometry.orbit(), dem, x, wvl,
                        out_geometry.lookSide(), r2g_params.tol_height,
                        r2g_params.look_min, r2g_params.look_max);

                llh = ellipsoid.xyzToLonLat(x);

                if (height != nullptr) {
                    height[j * out_geometry.gridWidth() + i] = llh[2];
                }
                if (not converged) {
                    all_converged = false;
                    out[j * out_geometry.gridWidth() + i] = {nan, nan};
                    if (height != nullptr) {
                        height[j * out_geometry.gridWidth() + i] = nan;
                    }
                    continue;
                }
            }

            // run geo2rdr using input data's orbit and azimuth carrier to
            // estimate the center of the coherent processing window for the
            // target
            double t, r;
            {
                auto converged =
                        geo2rdr_bracket(x, in_geometry.orbit(),
                                in_geometry.doppler(), t, r, wvl,
                                in_geometry.lookSide(), g2r_params.tol_aztime,
                                g2r_params.time_start, g2r_params.time_end);

                if (not converged) {
                    all_converged = false;
                    out[j * out_geometry.gridWidth() + i] = {nan, nan};
                    continue;
                }
            }

            // get platform position and velocity at center of CPI
            Vec3 p, v;
            in_geometry.orbit().interpolate(&p, &v, t);

            // estimate synthetic aperture length required to achieve the
            // desired azimuth resolution
            double l = wvl * r * (p.norm() / x.norm()) / (2. * ds);

            // approximate CPI duration (assuming constant platform velocity)
            double cpi = l / v.norm();

            // get coherent integration bounds (pulse indices)
            double tstart = t - 0.5 * cpi;
            double tstop = t + 0.5 * cpi;
            double t0 = in_azimuth_time.first();
            double dt = in_azimuth_time.spacing();
            auto kstart = static_cast<int>(std::floor((tstart - t0) / dt));
            auto kstop = static_cast<int>(std::ceil((tstop - t0) / dt));
            kstart = std::max(kstart, 0);
            kstop = std::min(kstop, in_azimuth_time.size());

            // estimate dry troposphere delay
            double tau_atm = 0.;
            if (dry_tropo_model == DryTroposphereModel::TSX) {
                tau_atm = dryTropoDelayTSX(p, llh, ellipsoid);
            }

            // integrate pulses
            out[j * out_geometry.gridWidth() + i] =
                    sumCoherent(in, sampling_window, pos, vel, x, fc, tau_atm,
                                kernel, kstart, kstop);
        }
    }

    if (not all_converged) {
        return ErrorCode::FailedToConverge;
    }
    return ErrorCode::Success;
}


static Vec3 vector_mean(const std::vector<Vec3>& vecs)
{
    Vec3 sum = {0, 0, 0};
    for (const auto& vec : vecs) {
        sum += vec;
    }
    return sum * (1.0 / vecs.size());
}

double
getPolarAngleTimeConstant(const double fc, const double vs,
        const double bandwidth, const double c)
{
    // Yegulalp, Eq. (11)
    const auto fmax = fc + bandwidth / 2;
    return c / (2 * fmax * vs);
}


std::tuple<PolarGrid, std::vector<Vec3>, std::vector<Vec3>>
setupPolarGridForPulses(
        const RadarGeometry& in_geometry,
        const Eigen::Ref<const Eigen::VectorXd>& azimuth_time,
        double range_bandwidth,
        double azimuth_resolution,
        double oversample_range, double oversample_azimuth,
        int num_doppler_eval,
        bool densify_for_fast_transforms)
{
    // Interpolate platform position & velocity at each pulse
    const auto nt = azimuth_time.size();
    std::vector<Vec3> pos(nt), vel(nt);

    for (auto i = decltype(nt){0}; i < nt; ++i) {
        double t = azimuth_time[i];
        in_geometry.orbit().interpolate(&pos[i], &vel[i], t);
    }

    // Use mean position as origin of polar grid.
    const Vec3 origin = vector_mean(pos);

    // For the along-track axis we could fit a line to the positions, or use the
    // dominant eigenvector of the position sample covariance.  But the average
    // velocity is probably about the same and simpler to compute.
    Vec3 axis = vector_mean(vel);
    const auto vs = axis.norm();
    axis *= 1.0 / vs;

    constexpr auto c = isce3::core::speed_of_light;
    const auto fc = c / in_geometry.wavelength();
    const auto slant_range = in_geometry.slantRange();

    // Yegulalp, Eq. (11) and (12)
    const auto tq = getPolarAngleTimeConstant(fc, vs, range_bandwidth, c);
    const auto duration = azimuth_time[nt - 1] - azimuth_time[0];
    auto dq = tq / (duration * oversample_azimuth);
    auto dr = c / (2 * range_bandwidth * oversample_range);

    // Though inefficient, user might try to combine more pulses than are
    // needed to achieve the desired azimuth resolution.  For example, they
    // might try to backproject all pulses from a stripmap radar in one shot.
    const auto dq_min = azimuth_resolution /
        (slant_range.last() * oversample_azimuth);
    if (dq < dq_min) {
        // TODO emit a warning?
        dq = dq_min;
    }

    // Our polar data structures use a constant Doppler centroid (DC) vs range.
    // If we have some DC variation over the swath, we'll increase the Doppler
    // bandwidth enough to accommodate it.  Later we can mask out the pixels
    // outside the desired azimuth band if desired.
    // We will assume the DC is stable over the slow-time span of the pulses.
    const auto
        tmid = (azimuth_time[0] + azimuth_time[nt - 1]) / 2,
        r0 = slant_range.first(),
        r1 = slant_range.last(),
        dop2q = c / (fc * 2 * vs);
    auto q0 = in_geometry.doppler().eval(tmid, r0) * dop2q;
    auto q1 = q0;
    for (int i = 1; i < num_doppler_eval; ++i) {
        const auto ri = r0 + i * (r1 - r0) / (num_doppler_eval - 1);
        const auto qi = in_geometry.doppler().eval(tmid, ri) * dop2q;
        q0 = std::min(q0, qi);
        q1 = std::max(q1, qi);
    }
    auto qmid = (q0 + q1) / 2;
    auto qspan = (q1 - q0) + c / (fc * 2 * azimuth_resolution);

    int nr = 1 + static_cast<int>(std::ceil((r1 - r0) / dr));
    int nq = 1 + static_cast<int>(std::ceil(qspan / dq));

    // adjust spacing so we end up with a fast FFT sizes
    if (densify_for_fast_transforms) {
        nr = nextFastPower(nr);
        nq = nextFastPower(nq);
        dr = (r1 - r0) / nr;
        dq = qspan / nq;  // possibly smaller than dq_min
    }

    auto pgrid = PolarGrid{azimuth_time[0], azimuth_time[nt - 1],
        origin, axis, Linspace<double>(r0, dr, nr),
        Linspace<double>(qmid - dq * (nq - 1) / 2, dq, nq),
        in_geometry.lookSide()};

    return {pgrid, pos, vel};
}


std::tuple<ErrorCode, PolarGrid, std::unique_ptr<std::complex<float>[]>, std::unique_ptr<float[]>>
backprojectFirstStage(
        const std::complex<float>* in, const RadarGeometry& in_geometry,
        const Eigen::Ref<const Eigen::VectorXd>& in_azimuth_time,
        double range_bandwidth,
        const DEMInterpolator& dem, double fc, double ds,
        const Kernel<float>& kernel, DryTroposphereModel dry_tropo_model,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
        double oversample_range, double oversample_azimuth)
{
    using isce3::geometry::detail::polar2geo_bracket;

    if (in_azimuth_time.size() < 2) {
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "require at least two pulses in initial FBP stage");
    }

    static constexpr double c = isce3::core::speed_of_light;
    static constexpr auto nan = std::numeric_limits<float>::quiet_NaN();

    // check that dry_tropo_model is supported internally
    if (not(dry_tropo_model == DryTroposphereModel::NoDelay or
            dry_tropo_model == DryTroposphereModel::TSX)) {

        std::string errmsg = "unexpected dry troposphere model";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    // awful hacks for clang https://godbolt.org/z/6rrThhK3W
    PolarGrid out_grid {0.0, 0.0, {0,0,0}, {1,0,0}, {}, {}, {}};
    std::vector<Vec3> pos, vel;
    std::tie(out_grid, pos, vel) = setupPolarGridForPulses(in_geometry,
        in_azimuth_time,
        range_bandwidth, ds, oversample_range,
        oversample_azimuth, 2, true);

    const auto npix = static_cast<size_t>(out_grid.length()) * out_grid.width();
    auto height = std::make_unique<float[]>(npix);
    auto out = std::make_unique<std::complex<float>[]>(npix);

    // range sampling window
    auto in_slant_range = in_geometry.slantRange();
    double swst = 2. * in_slant_range.first() / c;
    double dtau = 2. * in_slant_range.spacing() / c;
    int nr = in_slant_range.size();
    Linspace<double> sampling_window(swst, dtau, nr);

    // reference ellipsoid
    int epsg = dem.epsgCode();
    const Ellipsoid ellipsoid = makeProjection(epsg)->ellipsoid();

    // loop over targets in output grid
    bool all_converged = true;
#pragma omp parallel for
    for (int j = 0; j < out_grid.sin_squint.size(); ++j) {
        const double
            q = out_grid.sin_squint[j],
            c = std::sqrt(1.0 - q * q);
        for (int i = 0; i < out_grid.range.size(); ++i) {

            // Run polar2geo to get target position.
            // Only need LLH if dumping height or using TSX atmosphere model,
            // but just compute it unconditionally.
            Vec3 x, llh;
            {
                const double r = out_grid.range[i];
                double look_angle;

                const auto status = polar2geo_bracket(&x, &look_angle,
                        out_grid.origin, out_grid.axis, r, q, c, dem, ellipsoid,
                        in_geometry.lookSide(), r2g_params);

                llh = ellipsoid.xyzToLonLat(x);
                height[j * out_grid.width() + i] = llh[2];

                if (status != isce3::error::ErrorCode::Success) {
                    all_converged = false;
                    out[j * out_grid.width() + i] = {nan, nan};
                    height[j * out_grid.width() + i] = nan;
                    continue;
                }
            }

            // estimate dry troposphere delay
            double tau_atm = 0.;
            if (dry_tropo_model == DryTroposphereModel::TSX) {
                tau_atm = dryTropoDelayTSX(out_grid.origin, llh, ellipsoid);
            }

            // TODO range-dependent Doppler mask?
            int kstart = 0, kstop = in_geometry.gridLength();

            // integrate pulses
            out[j * out_grid.width() + i] =
                    sumCoherent(in, sampling_window, pos, vel, x, fc, tau_atm,
                                kernel, kstart, kstop);
        }
    }

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

    auto status =
            all_converged ? ErrorCode::Success : ErrorCode::FailedToConverge;
    return std::make_tuple(status, out_grid, std::move(out), std::move(height));
}

PolarGrid
mergePolarGrids(const std::vector<PolarGrid>& grids,
    const DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
    const std::optional<double>& dq_min,
    const std::optional<double>& tq)
{
    if (grids.size() <= 0) {
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "can't find common grid among empty list");
    } else if (grids.size() == 1) {
        return grids[0];
    }

    // reference ellipsoid
    Ellipsoid ellipsoid = makeProjection(dem.epsgCode())->ellipsoid();

    // Compute a bunch of stats with a first pass over the data.
    // Average origin and axis, weighted by aperture duration.
    Vec3 origin{0, 0, 0}, axis{0, 0, 0};
    // Inferred dimensionless Doppler spacing time constant
    double tq_inferred = 0.0;
    // Min range spacing (in case different among grids)
    auto dr = grids[0].range.spacing();
    // Need total aperture size and sum of subaperture sizes.
    // These are not equal if there are gaps or overlap between subapertures.
    auto t_min = grids[0].aztime_start;  // assume start > end
    auto t_max = grids[0].aztime_end;  // assume start > end
    double sum_durations = 0;
    const auto look_side = grids[0].look_side;

    for (const auto& grid : grids) {
        const auto duration = grid.aztime_end - grid.aztime_start;  // + PRI ??
        sum_durations += duration;
        t_min = std::min(t_min, grid.aztime_start);  // assume start > end
        t_max = std::max(t_max, grid.aztime_end);  // assume start > end
        dr = std::min(dr, grid.range.spacing());
        origin += duration * grid.origin;
        axis += duration * grid.axis;
        tq_inferred += duration * (grid.sin_squint.spacing() * duration);
        if (grid.look_side != look_side) {
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "inconsistent look_side among input polar grids");
        }
    }
    origin *= 1.0 / sum_durations;
    axis *= 1.0 / axis.norm();
    tq_inferred /= sum_durations;

    // In general, figuring out the required Doppler spacing is pretty complex.
    // You'd want to figure out the Doppler bandwidth observed by all targets
    // across all grids, maxing out around the azimuth resolution.
    // For now let's just just be conservative and increase it linearly.
    auto dq = tq.value_or(tq_inferred) / (t_max - t_min);

    // But the user can override this.
    if (dq_min) {
        dq = std::max(dq_min.value(), dq);
    }

    // Compute range & Doppler bounds of new grid using corners of each input.
    // Use lambda to avoid copy/paste.
    using isce3::geometry::detail::polar2polar_bracket;
    auto polar2polar = [&](const PolarGrid& grid, double r, double ssq) {
        auto csq = std::sqrt(1.0 - ssq * ssq);
        double r_out, ssq_out;
        auto ec = polar2polar_bracket(&ssq_out, &r_out, ssq, csq, r,
            grid.origin, grid.axis, origin, axis, dem, ellipsoid, look_side,
            r2g_params);
        if (ec != ErrorCode::Success) {
            throw isce3::except::DomainError(ISCE_SRCINFO(),
                "polar2polar failed with ErrorCode (" +
                isce3::error::getErrorString(ec) + ") for point at r="
                + std::to_string(r) + " sin_squint=" + std::to_string(ssq));
        }
        return std::make_tuple(r_out, ssq_out);
    };

    // TODO We're working with pixel centers.  Probably we should match grid
    // boundaries and throw a bunch of dx/2 terms around.  The Doppler spacing,
    // especially, will be different.
    auto [r_min, q_min] = polar2polar(grids[0], grids[0].range[0], grids[0].sin_squint[0]);
    auto r_max = r_min, q_max = q_min;
    for (const auto& grid : grids) {
        for (const auto& ri : {grid.range.first(), grid.range.last()}) {
            for (const auto& qi : {grid.sin_squint.first(), grid.sin_squint.last()}) {
                const auto [ro, qo] = polar2polar(grid, ri, qi);
                r_min = std::min(r_min, ro);
                r_max = std::max(r_max, ro);
                q_min = std::min(q_min, qo);
                q_max = std::max(q_max, qo);
            }
        }
    }

    int nr = 1 + static_cast<int>(std::round((r_max - r_min) / dr));
    int nq = 1 + static_cast<int>(std::round((q_max - q_min) / dq));

    // TODO The ceil() means potentially extra data.  It might be preferable to
    // pad equally on both sides, rather than adding all the extra to the end.
    return PolarGrid{t_min, t_max, origin, axis,
        Linspace<double>(r_min, dr, nr),
        Linspace<double>(q_min, dq, nq),
        look_side};
}


void mergePolarImages(
    const std::vector<PolarGrid>& grids,
    const std::vector<NFFT2d<float>>& image_interpolators,
    const PolarGrid& output_grid,
    Eigen::Ref<isce3::core::EArray2D<std::complex<float>>> output_image,
    const double fc,
    const DEMInterpolator& dem,
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
    if (az_block_size < 0) {
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "azimuth block size must be positive");
    }
    az_block_size = std::min(az_block_size, output_grid.sin_squint.size());

    auto block_positions = isce3::core::EArray2D<Vec3>();
    block_positions.resize(az_block_size, output_grid.width());

    // reference ellipsoid
    Ellipsoid ellipsoid = makeProjection(dem.epsgCode())->ellipsoid();

    // wavenumber
    const double kw = 4 * M_PI * fc / isce3::core::speed_of_light;

    using isce3::geometry::detail::polar2polar_bracket;
    using isce3::geometry::geo2polar;

    // loop over output blocks
    auto n_blocks = (m + az_block_size - 1) / az_block_size;
    for (auto i_block = decltype(n_blocks){0}; i_block < n_blocks; ++i_block) {
        auto i_row0 = i_block * az_block_size;
        auto i_row1 = std::min(i_row0 + az_block_size, m);

        // Compute output pixel 3D locations
        using isce3::geometry::detail::polar2geo_bracket;
        #pragma omp parallel for collapse(2)
        for (auto i_row = i_row0; i_row < i_row1; ++i_row) {
            for (auto j = decltype(n){0}; j < n; ++j) {
                auto i = i_row - i_row0;
                double look_angle;
                const auto ssq = output_grid.sin_squint[i_row];
                const auto csq = std::sqrt(1.0 - ssq * ssq);
                auto ec = polar2geo_bracket(&block_positions(i, j), &look_angle,
                    output_grid.origin, output_grid.axis, output_grid.range[j],
                    ssq, csq, dem, ellipsoid, output_grid.look_side, r2g_params);
                if (ec != ErrorCode::Success) {
                    throw isce3::except::DomainError(ISCE_SRCINFO(),
                        "polar2geo failed with ErrorCode (" +
                        isce3::error::getErrorString(ec) + ") for point at r="
                        + std::to_string(output_grid.range[j]) + " sin_squint="
                        + std::to_string(ssq));
                } // err
            } // columns
        } // rows

        // loop over input images
        for (auto i_img = decltype(num_images){0}; i_img < num_images; ++i_img) {
            const auto& input_grid = grids[i_img];
            const auto& nfft = image_interpolators[i_img];
            const auto npix = static_cast<size_t>(n) * (i_row1 - i_row0);
            auto ec = projectPolarToGeo(output_image.row(i_row0).data(),
                block_positions.data(), npix, input_grid, nfft, kw);
            if (ec != ErrorCode::Success) {
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                    "projectPolarToGeo failed with ErrorCode (" +
                    isce3::error::getErrorString(ec) + ")");
            } // error
        } // images
    } // blocks

    // Baseband.  Note that we could do this at the same time as the
    // reprojection but it'd require a fair bit of copy/paste.
    Eigen::VectorXcf phasors(n);
    #pragma omp parallel for
    for (auto j = decltype(n){0}; j < n; ++j) {
        const double arg = -kw * output_grid.range[j];
        phasors(j) = std::complex<float>(std::cos(arg), std::sin(arg));
    }
    #pragma omp parallel for collapse(2)
    for (auto i = decltype(m){0}; i < m; ++i) {
        for (auto j = decltype(n){0}; j < n; ++j) {
            output_image(i, j) *= phasors(j);
        } // columns
    } // rows
}


// For now structure like backproject() with inner loop on target.
// Might make more sense to project on image at a time instead.
ErrorCode
backprojectFinalStage(std::complex<float>* out,
        const RadarGeometry& out_geometry,
        const isce3::core::Orbit& in_orbit,
        const isce3::core::LUT2d<double>& in_doppler,
        const std::vector<PolarGrid>& grids,
        const std::vector<NFFT2d<float>>& image_interpolators,
        const DEMInterpolator& dem, double fc, double ds,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
        const isce3::geometry::detail::Geo2RdrBracketParams& g2r_params,
        float* height)
{
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

    const size_t nout = out_geometry.gridLength() * out_geometry.gridWidth();
    std::vector<Vec3> x(nout);
    std::vector<double> tstart(nout), tend(nout);

    // loop over targets in output grid
    bool all_converged = true;
    #pragma omp parallel for
    for (size_t iflat = 0; iflat < nout; ++iflat) {
        const size_t j = iflat / out_slant_range.size();
        const size_t i = iflat % out_slant_range.size();

        // Run rdr2geo using orbit and Doppler associated with output grid
        // to get target position.  Only need LLH if dumping height or
        // using TSX atmosphere model, but just compute it unconditionally.
        Vec3 llh;
        {
            double t = out_azimuth_time[j];
            double r = out_slant_range[i];
            double fD = out_geometry.doppler().eval(t, r);

            const int converged = rdr2geo_bracket(t, r, fD,
                    out_geometry.orbit(), dem, x[iflat], wvl,
                    out_geometry.lookSide(), r2g_params.tol_height,
                    r2g_params.look_min, r2g_params.look_max);

            llh = ellipsoid.xyzToLonLat(x[iflat]);

            if (height != nullptr) {
                height[iflat] = llh[2];
            }
            if (not converged) {
                all_converged = false;
                out[iflat] = {nan, nan};
                if (height != nullptr) {
                    height[iflat] = nan;
                }
                continue;
            }
        }

        // run geo2rdr to estimate the center of the coherent processing
        // window for the target
        double t, r;
        {
            auto converged =
                    geo2rdr_bracket(x[iflat], in_orbit,
                            in_doppler, t, r, wvl,
                            out_geometry.lookSide(),  // assumed same side
                            g2r_params.tol_aztime,
                            g2r_params.time_start, g2r_params.time_end);

            if (not converged) {
                all_converged = false;
                out[iflat] = {nan, nan};
                continue;
            }
        }

        // get platform position and velocity at center of CPI
        Vec3 p, v;
        in_orbit.interpolate(&p, &v, t);

        // estimate synthetic aperture length required to achieve the
        // desired azimuth resolution
        double l = wvl * r * (p.norm() / x[iflat].norm()) / (2. * ds);

        // approximate CPI duration (assuming constant platform velocity)
        double cpi = l / v.norm();

        // get coherent integration bounds (pulse indices)
        tstart[iflat] = t - cpi / 2;
        tend[iflat] = tstart[iflat] + cpi;
    }

    // TODO reduce tstart & tend
    // TODO check this O(log(n)) algorithm
    //const auto kstart = std::distance(ends.begin(),
    //    std::lower_bound(ends.begin(), ends.end(), tstart));
    //const auto kstop = std::distance(starts.begin(),
    //    std::upper_bound(starts.start(), starts.end(), tstart + cpi));
    const auto num_images = image_interpolators.size();
    const decltype(num_images) kstart = 0, kstop = num_images;

    for (auto k = kstart; k < kstop; ++k) {
        // check if we need to replan FFTs
        const auto& grid = grids[k];
        const auto& nfft = image_interpolators[k];

        #pragma omp parallel for
        for (size_t iflat = 0; iflat < nout; ++iflat) {
            // check if target seen in this subimage
            if ((grid.aztime_end < tstart[iflat]) or (grid.aztime_start > tend[iflat])) {
                continue;
            }
            // compute target location in polar grid
            double sin_squint, range;
            geo2polar(&sin_squint, &range, x[iflat], grid.origin, grid.axis);
            // convert to image index
            const double ix = (range - grid.range.first()) / grid.range.spacing(),
                iy = (sin_squint - grid.sin_squint.first()) / grid.sin_squint.spacing();
            // interpolate baseband data
            const auto z = nfft.interp({iy, ix}, false);
            // compensate phase and sum contribution
            const double phase = kw * range;
            out[iflat] +=
                z * std::complex<float>(std::cos(phase), std::sin(phase));
        }
    }

    if (not all_converged) {
        return ErrorCode::FailedToConverge;
    }
    return ErrorCode::Success;
}

// WIP stuff to do one polar image at a time.

ErrorCode
projectPolarToGeo(
        std::complex<float>* geo_image,
        const Vec3* geo_points,
        const size_t n,
        const PolarGrid& grid,
        const NFFT2d<float>& nfft,
        const double kw)
{
    #pragma omp parallel for
    for (size_t i= 0; i < n; ++i) {
        // compute target location in polar grid
        double sin_squint, range;
        geo2polar(&sin_squint, &range, geo_points[i], grid.origin, grid.axis);
        // convert to image index
        const double ix = (range - grid.range.first()) / grid.range.spacing(),
            iy = (sin_squint - grid.sin_squint.first()) / grid.sin_squint.spacing();
        // interpolate baseband data
        const auto z = nfft.interp({iy, ix}, /* periodic */ false);
        // compensate phase and sum contribution
        const double phase = kw * range;
        geo_image[i] +=
            z * std::complex<float>(std::cos(phase), std::sin(phase));
    }
    return ErrorCode::Success;
}

std::tuple<double, double, double, double, isce3::error::ErrorCode>
findPolarGridBoundingBoxInRadarCoord(
    const PolarGrid& polar_grid,
    const Orbit& orbit,
    const LUT2d<double>& doppler,
    const double wavelength,
    const LookSide lookside,
    const DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
    const isce3::geometry::detail::Geo2RdrBracketParams& g2r_params,
    const int nextra)
{
    using isce3::geometry::detail::polar2geo_bracket;
    if (nextra < 0) {
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "specified negative number of extra points");
    }

    // get (angle, range) points along perimeter of polar grid
    const int n = 4 * (1 + nextra);
    int nwritten = 0;
    std::vector<std::array<double, 2>> points(n);
    for (int i = 0; i <= nextra; ++i) {
        const auto q = polar_grid.sin_squint.first();
        const auto dr = (polar_grid.range.last() - polar_grid.range.first()) /
            (1 + nextra);
        const auto r = polar_grid.range.first() + i * dr;
        points[nwritten++] = {q, r};
    }
    for (int i = 0; i <= nextra; ++i) {
        const auto r = polar_grid.range.last();
        const auto dq = (polar_grid.sin_squint.last() - polar_grid.sin_squint.first()) /
            (1 + nextra);
        const auto q = polar_grid.sin_squint.first() + i * dq;
        points[nwritten++] = {q, r};
    }
    for (int i = 0; i <= nextra; ++i) {
        const auto q = polar_grid.sin_squint.last();
        const auto dr = (polar_grid.range.last() - polar_grid.range.first()) /
            (1 + nextra);
        const auto r = polar_grid.range.last() - i * dr;
        points[nwritten++] = {q, r};
    }
    for (int i = 0; i <= nextra; ++i) {
        const auto r = polar_grid.range.first();
        const auto dq = (polar_grid.sin_squint.last() - polar_grid.sin_squint.first()) /
            (1 + nextra);
        const auto q = polar_grid.sin_squint.last() - i * dq;
        points[nwritten++] = {q, r};
    }
    assert(nwritten == n);

    int epsg = dem.epsgCode();
    Ellipsoid ellipsoid = makeProjection(epsg)->ellipsoid();
    auto status = ErrorCode::Success;

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        // read polar coordinate
        const double ssq = points[i][0];
        const double rin = points[i][1];
        // compute cos from sin assuming abs(squint) < 90 deg
        const double csq = std::sqrt(1.0 - ssq * ssq);
        // convert to xyz
        Vec3 xyz;
        double lookangle;
        auto err = polar2geo_bracket(&xyz, &lookangle, polar_grid.origin,
            polar_grid.axis, rin, ssq, csq, dem, ellipsoid,
            lookside, r2g_params);
        if (err != ErrorCode::Success) {
            status = err;
        }
        // convert to stripmap radar coordinates
        double tout, rout;
        int success = geo2rdr_bracket(xyz, orbit,
            doppler, tout, rout, wavelength,
            lookside, g2r_params.tol_aztime, g2r_params.time_start,
            g2r_params.time_end);
        if (!success) {
            status = ErrorCode::FailedToConverge;
        }
        // write back
        points[i] = {tout, rout};
    }

    // find extrema
    double tmin, tmax, rmin, rmax;
    tmin = tmax = points[0][0];
    rmin = rmax = points[0][1];
    for (int i = 1; i < n; ++i) {
        const double t = points[i][0], r = points[i][1];
        if (t > tmax) tmax = t;
        if (t < tmin) tmin = t;
        if (r > rmax) rmax = r;
        if (r < rmin) rmin = r;
    }

    return std::tie(tmin, tmax, rmin, rmax, status);
}

std::tuple<isce3::product::RadarGridParameters, isce3::error::ErrorCode>
findPolarGridBoundingBoxInRadarGrid(
    const PolarGrid& polar_grid,
    const RadarGeometry& radar_geom,
    const DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
    const isce3::geometry::detail::Geo2RdrBracketParams& g2r_params,
    const int nextra)
{
    auto [tmin, tmax, rmin, rmax, status] =
        findPolarGridBoundingBoxInRadarCoord(polar_grid, radar_geom.orbit(),
            radar_geom.doppler(), radar_geom.wavelength(),
            radar_geom.lookSide(), dem, r2g_params, g2r_params, nextra);

    // too much typing
    const auto t0 = radar_geom.sensingTime().first();
    const auto dt = radar_geom.sensingTime().spacing();
    const auto r0 = radar_geom.slantRange().first();
    const auto dr = radar_geom.slantRange().spacing();
    const int m = static_cast<int>(radar_geom.gridLength());
    const int n = static_cast<int>(radar_geom.gridWidth());

    // convert extrema to indices in radar grid
    int i0, j0, i1, j1;
    i0 = static_cast<int>(std::floor((tmin - t0) / dt));
    i1 = static_cast<int>(std::ceil((tmax - t0) / dt));
    j0 = static_cast<int>(std::floor((rmin - r0) / dr));
    j1 = static_cast<int>(std::ceil((rmax - r0) / dr));

    // copy of radar grid but with shape = (0, 0)
    using isce3::product::RadarGridParameters;
    const auto& igrid = radar_geom.radarGrid();
    const auto empty =  RadarGridParameters(t0, igrid.wavelength(),
        igrid.prf(), r0, dr, igrid.lookSide(), 0, 0, igrid.refEpoch());

    // return empty grid if non-overlapping
    if ((i1 < 0) or (i0 >= m) or (j1 < 0) or (j0 >= n)) {
        return std::tie(empty, status);
    }

    // otherwise clamp to grid bounds
    i0 = std::max(0, std::min(i0, m - 1));
    i1 = std::max(0, std::min(i1, m));
    j0 = std::max(0, std::min(j0, n - 1));
    j1 = std::max(0, std::min(j1, n));

    const auto ogrid =  RadarGridParameters(
        radar_geom.sensingTime()[i0],
        igrid.wavelength(),
        igrid.prf(),
        radar_geom.slantRange()[j0],
        igrid.rangePixelSpacing(),
        igrid.lookSide(),
        i1 - i0,
        j1 - j0,
        igrid.refEpoch());

    return std::tie(ogrid, status);
}

std::tuple<std::vector<Vec3>, ErrorCode>
computeRadarGridGeoPoints(
    const RadarGeometry& geom,
    const DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params)
{
    const size_t n = geom.gridLength() * geom.gridWidth();
    std::vector<Vec3> points(n);
    auto status = computeRadarGridGeoPoints(points.data(), geom, dem, r2g_params);
    return std::tie(points, status);
}

ErrorCode
computeRadarGridGeoPoints(
    Vec3* points,
    const RadarGeometry& geom,
    const DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params)
{
    const size_t n = geom.gridLength() * geom.gridWidth();
    ErrorCode status = ErrorCode::Success;
    #pragma omp parallel for
    for (size_t k = 0; k < n; ++k) {
        const int i = static_cast<int>(k / geom.gridWidth());
        const int j = static_cast<int>(k % geom.gridWidth());
        const double t = geom.sensingTime()[i];
        const double r = geom.slantRange()[j];
        const double fd = geom.doppler().eval(t, r);
        const int success = isce3::geometry::rdr2geo_bracket(t, r, fd,
            geom.orbit(), dem, points[k], geom.wavelength(), geom.lookSide(),
            r2g_params.tol_height, r2g_params.look_min, r2g_params.look_max);
        if (!success) {
            // race condition okay since always pushing the same value
            status = ErrorCode::FailedToConverge;
        }
    }
    return status;
}

} // namespace focus
} // namespace isce3
