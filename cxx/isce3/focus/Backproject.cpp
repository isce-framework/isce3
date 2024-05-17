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
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/geometry.h>
#include <isce3/geometry/rdr2geo_roots.h>
#include <isce3/geometry/geo2rdr_roots.h>
#include <limits>
#include <string>
#include <vector>

#include "BistaticDelay.h"

using namespace isce3::core;
using namespace isce3::geometry;
using isce3::error::ErrorCode;

using isce3::container::RadarGeometry;

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


std::tuple<ErrorCode, PolarGrid, std::unique_ptr<std::complex<float>[]>, std::unique_ptr<float[]>>
backprojectFirstStage(
        const std::complex<float>* in, const RadarGeometry& in_geometry,
        const std::vector<double>& in_azimuth_time,
        double range_bandwidth,
        const DEMInterpolator& dem, double fc, double ds,
        const Kernel<float>& kernel, DryTroposphereModel dry_tropo_model,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
        double oversample_range, double oversample_azimuth)
{
    using isce3::geometry::detail::polar2geo_bracket;

    if (in_azimuth_time.size() < 2) {
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "require at least two pulses in FBP stage1");
    }

    static constexpr double c = isce3::core::speed_of_light;
    static constexpr auto nan = std::numeric_limits<float>::quiet_NaN();

    // check that dry_tropo_model is supported internally
    if (not(dry_tropo_model == DryTroposphereModel::NoDelay or
            dry_tropo_model == DryTroposphereModel::TSX)) {

        std::string errmsg = "unexpected dry troposphere model";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    // get input & output radar grid azimuth time & slant range
    Linspace<double> in_slant_range = in_geometry.slantRange();

    // interpolate platform position & velocity at each pulse
    std::vector<Vec3> pos(in_azimuth_time.size());
    std::vector<Vec3> vel(in_azimuth_time.size());
    for (int i = 0; i < in_azimuth_time.size(); ++i) {
        double t = in_azimuth_time[i];
        in_geometry.orbit().interpolate(&pos[i], &vel[i], t);
    }

    const PolarGrid out_grid = [&](void) {
        const auto iend = in_azimuth_time.size() - 1;
        const auto jend = in_slant_range.size() - 1;

        const Vec3 origin = (pos[0] + pos[iend]) / 2;
        Vec3 axis = (vel[0] + vel[iend]) / 2;
        const double vs = axis.norm();
        axis /= vs;

        const double
            fmax = fc + range_bandwidth / 2,
            length = vs * (in_azimuth_time[iend] - in_azimuth_time[0]),
            // Yegulalp, Eq. (11) and (12)
            dq = c / (2 * fmax * length * oversample_azimuth),
            dr = c / (2 * range_bandwidth * oversample_range);

        // evaluate Doppler at a couple of points to try to cover variation
        const double
            tmid = (in_azimuth_time[0] + in_azimuth_time[iend]) / 2,
            r0 = in_slant_range.first(),
            r1 = in_slant_range[jend],
            dop2q = c / (fc * 2 * vs),
            q0 = in_geometry.doppler().eval(tmid, r0) * dop2q,
            q1 = in_geometry.doppler().eval(tmid, r1) * dop2q,
            qmid = (q0 + q1) / 2,
            qspan = std::abs(q1 - q0) + c / (fc * 2 * ds);

        const int nr = static_cast<int>(std::ceil((r1 - r0) / dr));
        const int nq = static_cast<int>(std::ceil(qspan / dq));
        return PolarGrid{in_azimuth_time[0], in_azimuth_time[iend],
            origin, axis, Linspace<double>(r0, dr, nr),
            Linspace<double>(qmid - qspan / 2, dq, nq)};
    }();

    const auto npix = out_grid.length() * out_grid.width();
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

            // integrate pulses
            out[j * out_grid.width() + i] =
                    sumCoherent(in, sampling_window, pos, vel, x, fc, tau_atm,
                                kernel, 0, in_geometry.gridLength());
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


// For now structure like backproject() with inner loop on target.
// Might make more sense to project on image at a time instead.
ErrorCode
backprojectFinalStage(std::complex<float>* out,
        const RadarGeometry& out_geometry,
        const isce3::core::Orbit& in_orbit,
        const isce3::core::LUT2d<double>& in_doppler,
        const std::vector<PolarGrid>& grids,
        const std::vector<const std::complex<float>*>& images,
        const DEMInterpolator& dem, double fc, double ds,
        const Kernel<float>& kernel_rg, const Kernel<float>& kernel_az,
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

            // run geo2rdr to estimate the center of the coherent processing
            // window for the target
            double t, r;
            {
                auto converged =
                        geo2rdr_bracket(x, in_orbit,
                                in_doppler, t, r, wvl,
                                out_geometry.lookSide(),  // assumed same side
                                g2r_params.tol_aztime,
                                g2r_params.time_start, g2r_params.time_end);

                if (not converged) {
                    all_converged = false;
                    out[j * out_geometry.gridWidth() + i] = {nan, nan};
                    continue;
                }
            }

            // get platform position and velocity at center of CPI
            Vec3 p, v;
            in_orbit.interpolate(&p, &v, t);

            // estimate synthetic aperture length required to achieve the
            // desired azimuth resolution
            double l = wvl * r * (p.norm() / x.norm()) / (2. * ds);

            // approximate CPI duration (assuming constant platform velocity)
            double cpi = l / v.norm();

            // get coherent integration bounds (pulse indices)
            const auto tstart = t - cpi / 2, tend = tstart + cpi;
            // TODO check this O(log(n)) algorithm
            //const auto kstart = std::distance(ends.begin(),
            //    std::lower_bound(ends.begin(), ends.end(), tstart));
            //const auto kstop = std::distance(starts.begin(),
            //    std::upper_bound(starts.start(), starts.end(), tstart + cpi));
            const decltype(images.size()) kstart = 0, kstop = images.size();

            for (auto k = kstart; k < kstop; ++k) {
                const auto& grid = grids[k];
                // check if target seen in this subimage
                if ((grid.aztime_end < tstart) or (grid.aztime_start > tend)) {
                    continue;
                }
                // compute target location in polar grid
                double sin_squint, range;
                geo2polar(&sin_squint, &range, x, grid.origin, grid.axis);
                // convert to image index
                const double ix = (range - grid.range.first()) / grid.range.spacing(),
                    iy = (sin_squint - grid.sin_squint.first()) / grid.sin_squint.spacing();
                // interpolate baseband data
                const auto z = interp2d(kernel_rg, kernel_az, images[k], grid.width(),
                    /* stride x */ 1, grid.length(), /* stride y*/ grid.width(),
                    ix, iy);
                // compensate phase and sum contribution
                const double phase = kw * range;
                out[j * out_geometry.gridWidth() + i] +=
                    z * std::complex<float>(std::cos(phase), std::sin(phase));
            }
        }
    }

    if (not all_converged) {
        return ErrorCode::FailedToConverge;
    }
    return ErrorCode::Success;
}

} // namespace focus
} // namespace isce3
