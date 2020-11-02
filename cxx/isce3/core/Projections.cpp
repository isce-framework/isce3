#include "Projections.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace isce3 { namespace core {

int Geocent::forward(const Vec3& llh, Vec3& xyz) const
{
    // This is to transform LLH to Geocent, which is just a pass-through to
    // lonLatToXyz.
    ellipsoid().lonLatToXyz(llh, xyz);
    return 0;
}

int Geocent::inverse(const Vec3& xyz, Vec3& llh) const
{
    // This is to transform Geocent to LonLatHeight, which is just a
    // pass-through to xyzToLonLat.
    ellipsoid().xyzToLonLat(xyz, llh);
    return 0;
}

/**
 * @internal
 * Local function - Compute the real clenshaw summation. Also computes
 * Gaussian latitude for some B as clens(a, len(a), 2*B) + B.
 *
 * NOTE: The implementation here has been modified to allow for
 * encapsulating the gatg() implementation, as well as to make the
 * implementation details much clearer/cleaner.
 */
static double clens(const double* a, int size, double real)
{
    const double* p;
    double hr, hr1, hr2;
    for (p = a + size, hr2 = 0., hr1 = *(--p), hr = 0.; a - p;
         hr2 = hr1, hr1 = hr) {
        hr = -hr2 + (2. * hr1 * std::cos(real)) + *(--p);
    }
    return std::sin(real) * hr;
}

/**
 * @internal
 * Local function - Compute the complex clenshaw summation.
 *
 * NOTE: The implementation here has been modified to match the modified
 * implementation of the real clenshaw summation above. As expected with
 * complex->real behavior, if imag == 0, then I == 0 on return regardless of
 * other inputs (so maybe we just implement clenS(a,len(a),real,0,_,_) for
 * clens(a,len(a),real) to simplify the code space?)
 */
static double clenS(const double* a, int size, double real, double imag,
                    double& R, double& I)
{
    const double* p;
    double hr, hr1, hr2, hi, hi1, hi2;
    for (p = a + size, hr2 = 0., hi2 = 0., hi1 = 0., hr1 = *(--p), hi1 = 0.,
        hr = 0., hi = 0.;
         a - p; hr2 = hr1, hi2 = hi1, hr1 = hr, hi1 = hi) {
        hr = -hr2 + (2. * hr1 * std::cos(real) * std::cosh(imag)) -
             (-2. * hi1 * std::sin(real) * std::sinh(imag)) + *(--p);
        hi = -hi2 + (-2. * hr1 * std::sin(real) * std::sinh(imag)) +
             (2. * hi1 * std::cos(real) * std::cosh(imag));
    }

    // Bad practice - Should *either* modify R in-place *or* return R, not both.
    // I is modified, but not returned. Since R and I are tied, we should either
    // return a pair<,>(,) or modify in-place, not mix the strategies
    R = (std::sin(real) * std::cosh(imag) * hr) -
        (std::cos(real) * std::sinh(imag) * hi);
    I = (std::sin(real) * std::cosh(imag) * hi) +
        (std::cos(real) * std::sinh(imag) * hr);
    return R;
}

UTM::UTM(int code) : ProjectionBase(code)
{
    // Value constructor, delegates to base constructor before continuing with
    // UTM-specific setup code (previously contained in a private _setup()
    // method but moved given that _setup() was not supposed to be callable
    // after construction).
    if ((code > 32600) && (code <= 32660)) {
        zone = code - 32600;
        isnorth = true;
    } else if ((code > 32700) && (code <= 32760)) {
        zone = code - 32700;
        isnorth = false;
    } else {
        std::string errstr =
                "In UTM::UTM - Invalid EPSG Code for UTM Projection. Received ";
        errstr += std::to_string(code);
        errstr += ", expected in ranges (32600,32660] or (32700,32760].";
        throw std::invalid_argument(errstr);
    }

    lon0 = ((zone - 0.5) * (M_PI / 30.)) - M_PI;

    // Ellipsoid flattening
    double f = ellipsoid().e2() / (1. + std::sqrt(1 - ellipsoid().e2()));
    // Third flattening
    double n = f / (2. - f);

    // clang-format off

    // Gaussian -> Geodetic == cgb
    // Geodetic -> Gaussian == cbg
    cgb[0] = n * (2 + n * ((-2. / 3.) + n * (-2 + n * ((116. / 45.) + n *
            ((26. / 45.) + n * (-2854. / 675.))))));
    cbg[0] = n * (-2 + n * ((2. / 3.) + n * ((4. / 3.) + n * ((-82. / 45.) + n *
            ((32. / 45.) + n * (4642. / 4725.))))));
    cgb[1] = std::pow(n, 2) * ((7. / 3.) + n * ((-8. / 5.) + n *
            ((-227. / 45.) + n * ((2704. / 315.) + n * (2323. / 945.)))));
    cbg[1] = std::pow(n, 2) * ((5. / 3.) + n * ((-16. / 15.) + n *
            ((-13. / 9.) + n * ((904. / 315.) + n * (-1522. / 945.)))));
    cgb[2] = std::pow(n, 3) * ((56. / 15.) + n * ((-136. / 35.) + n *
            ((-1262. / 105.) + n * (73814. / 2835.))));
    cbg[2] = std::pow(n, 3) * ((-26. / 15.) + n * ((34. / 21.) + n *
            ((8. / 5.) + n * (-12686. / 2835.))));
    cgb[3] = std::pow(n, 4) * ((4279. / 630.) + n * ((-332. / 35.) + n *
            (-399572 / 14175.)));
    cbg[3] = std::pow(n, 4) * ((1237. / 630.) + n * ((-12. / 5.) + n *
            (-24832. / 14175.)));
    cgb[4] = std::pow(n, 5) * ((4174. / 315.) + n * (-144838. / 6237.));
    cbg[4] = std::pow(n, 5) * ((-734. / 315.) + n * (109598. / 31185.));
    cgb[5] = std::pow(n, 6) * (601676. / 22275.);
    cbg[5] = std::pow(n, 6) * (444337. / 155925.);

    // We have fixed k0 = 0.9996 here. This is standard for WGS84 zones. Proj4
    // allows this to be changed for custom definitions. We plan to support
    // standard definitions only.
    Qn = (0.9996 / (1. + n)) *
         (1. + n * n * ((1. / 4.) + n * n * ((1. / 64.) + ((n * n) / 256.))));

    // Elliptical N,E -> Spherical N,E == utg
    // Spherical N,E -> Elliptical N,E == gtu
    utg[0] = n * (-.5 + n * ((2. / 3.) + n * ((-37. / 96.) + n * ((1. / 360.) +
            n * ((81. / 512.) + n * (-96199. / 604800.))))));
    gtu[0] = n * (.5 + n * ((-2. / 3.) + n * ((5. / 16.) + n * ((41. / 180.) +
            n * ((-127. / 288.) + n * (7891. / 37800.))))));
    utg[1] = std::pow(n, 2) * ((-1. / 48.) + n * ((-1. / 15.) + n *
            ((437. / 1440.) + n * ((-46. / 105.) + n *
            (1118711. / 3870720.)))));
    gtu[1] = std::pow(n, 2) * ((13. / 48.) + n * ((-3. / 5.) + n *
            ((557. / 1440.) + n * ((281. / 630.) + n *
            (-1983433. / 1935360.)))));
    utg[2] = std::pow(n, 3) * ((-17. / 480.) + n * ((37. / 840.) + n *
            ((209. / 4480.) + n * (-5569. / 90720.))));
    gtu[2] = std::pow(n, 3) * ((61. / 240.) + n * ((-103. / 140.) + n *
            ((15061. / 26880.) + n * (167603. / 181440.))));
    utg[3] = std::pow(n, 4) * ((-4397. / 161280.) + n * ((11. / 504.) + n *
            (830251. / 7257600.)));
    gtu[3] = std::pow(n, 4) * ((49561. / 161280.) + n * ((-179. / 168.) + n *
            (6601661. / 7257600.)));
    utg[4] = std::pow(n, 5) * ((-4583. / 161280.) + n * (108847. / 3991680.));
    gtu[4] = std::pow(n, 5) * ((34729. / 80640.) + n * (-3418889. / 1995840.));
    utg[5] = std::pow(n, 6) * (-20648693. / 638668800.);
    gtu[5] = std::pow(n, 6) * (212378941. / 319334400.);

    // clang-format on

    // Gaussian latitude of origin latitude
    // JC - clens(_,_,0.) is always 0, should we hardcode/eliminate this?
    double Z = clens(cbg, 6, 0.);
    Zb = -Qn * (Z + clens(gtu, 6, 2 * Z));
}

int UTM::forward(const Vec3& llh, Vec3& utm) const
{
    // Elliptical Lat, Lon -> Gaussian Lat, Lon
    double gauss = clens(cbg, 6, 2. * llh[1]) + llh[1];
    // Adjust longitude for zone offset
    double lam = llh[0] - lon0;

    // Account for longitude and get Spherical N,E
    double Cn = std::atan2(std::sin(gauss), std::cos(lam) * std::cos(gauss));
    double Ce = std::atan2(
            std::sin(lam) * std::cos(gauss),
            std::hypot(std::sin(gauss), std::cos(gauss) * std::cos(lam)));

    // Spherical N,E to Elliptical N,E
    Ce = asinh(tan(Ce));
    double dCn, dCe;
    Cn += clenS(gtu, 6, 2 * Cn, 2 * Ce, dCn, dCe);
    Ce += dCe;

    if (std::fabs(Ce) <= 2.623395162778) {
        utm[0] = (Qn * Ce * ellipsoid().a()) + 500000.;
        utm[1] = (((Qn * Cn) + Zb) * ellipsoid().a()) +
                 (isnorth ? 0. : 10000000.);
        // UTM is lateral projection only, height is pass through.
        utm[2] = llh[2];
        return 0;
    } else {
        return 1;
    }
}

int UTM::inverse(const Vec3& utm, Vec3& llh) const
{
    double Cn = (utm[1] - (isnorth ? 0. : 10000000.)) / ellipsoid().a();
    double Ce = (utm[0] - 500000.) / ellipsoid().a();

    // Normalize N,E to Spherical N,E
    Cn = (Cn - Zb) / Qn;
    Ce /= Qn;

    if (std::fabs(Ce) <= 2.623395162778) {
        // N,E to Spherical Lat, Lon
        double dCn, dCe;
        Cn += clenS(utg, 6, 2 * Cn, 2 * Ce, dCn, dCe);
        Ce = std::atan(std::sinh(Ce + dCe));

        // Spherical Lat, Lon to Gaussian Lat, Lon
        double sinCe = std::sin(Ce);
        double cosCe = std::cos(Ce);
        Ce = std::atan2(sinCe, cosCe * std::cos(Cn));
        Cn = std::atan2(std::sin(Cn) * cosCe,
                        std::hypot(sinCe, cosCe * std::cos(Cn)));

        // Gaussian Lat, Lon to Elliptical Lat, Lon
        llh[0] = Ce + lon0;
        llh[1] = clens(cgb, 6, 2 * Cn) + Cn;
        // UTM is a lateral projection only. Height is pass through.
        llh[2] = utm[2];
        return 0;
    } else {
        return 1;
    }
}

/**
 * @internal
 * Local function - Determine small t from PROJ.4.
 */
static double pj_tsfn(double phi, double sinphi, double e)
{
    sinphi *= e;
    return tan(.5 * ((.5 * M_PI) - phi)) /
           std::pow((1. - sinphi) / (1. + sinphi), .5 * e);
}

PolarStereo::PolarStereo(int code) : ProjectionBase(code)
{
    // Set up various parameters for polar stereographic projection. Currently
    // only EPSG:3031 (Antarctic) and EPSG:3413 (Greenland) are supported.
    if (code == 3031) {
        isnorth = false;
        lat0 = -M_PI / 2.;
        // Only need absolute value
        lat_ts = (71. * M_PI) / 180.;
        lon0 = 0.;
    } else if (code == 3413) {
        isnorth = true;
        lat0 = M_PI / 2.;
        lat_ts = 70. * (M_PI / 180.);
        lon0 = -45. * (M_PI / 180.);
    } else {
        std::string errstr =
                "In PolarStereo::PolarStereo - Invalid EPSG Code for "
                "Polar Stereographic ";
        errstr += "projection. Received ";
        errstr += std::to_string(code);
        errstr += ", expected either 3031 (Antarctic) or 3413 (Greenland). "
                  "[NOTE: Other codes are ";
        errstr += "currently not supported]";
        throw std::invalid_argument(errstr);
    }
    e = std::sqrt(ellipsoid().e2());
    akm1 = std::cos(lat_ts) / pj_tsfn(lat_ts, std::sin(lat_ts), e);
    akm1 *= ellipsoid().a() /
            std::sqrt(1. - (std::pow(e, 2) * std::pow(std::sin(lat_ts), 2)));
}

int PolarStereo::forward(const Vec3& llh, Vec3& out) const
{
    double lam = llh[0] - lon0;
    double phi = llh[1] * (isnorth ? 1. : -1.);
    double temp = akm1 * pj_tsfn(phi, std::sin(phi), e);

    out[0] = temp * std::sin(lam);
    out[1] = -temp * std::cos(lam) * (isnorth ? 1. : -1.);
    // Height is just pass through
    out[2] = llh[2];

    return 0;
}

int PolarStereo::inverse(const Vec3& ups, Vec3& llh) const
{
    double tp = -std::hypot(ups[0], ups[1]) / akm1;
    double fact = (isnorth) ? 1 : -1;
    double phi_l = (.5 * M_PI) - (2. * std::atan(tp));

    double sinphi;
    double phi = 0.;
    for (int i = 8; i--; phi_l = phi) {
        sinphi = e * std::sin(phi_l);
        phi = 0.5 * M_PI +
              2. * std::atan(tp *
                             std::pow((1. + sinphi) / (1. - sinphi), -0.5 * e));
        if (std::fabs(phi_l - phi) < 1.e-10) {
            llh[0] = ((ups[0] == 0.) && (ups[1] == 0.))
                             ? 0.
                             : std::atan2(ups[0], -fact * ups[1]) + lon0;
            llh[1] = phi * fact;
            llh[2] = ups[2];
            return 0;
        }
    }
    return 1;
}

/**
 * @internal
 * Local function - ???
 */
static double pj_qsfn(double sinphi, double e, double one_es)
{
    double con = e * sinphi;
    return one_es * ((sinphi / (1. - std::pow(con, 2))) -
                     ((.5 / e) * std::log((1. - con) / (1. + con))));
}

CEA::CEA() : ProjectionBase(6933)
{
    // Set up parameters for equal area projection.
    lat_ts = M_PI / 6.;
    k0 = std::cos(lat_ts) /
         std::sqrt(1. - (ellipsoid().e2() * std::pow(std::sin(lat_ts), 2)));
    e = std::sqrt(ellipsoid().e2());
    one_es = 1. - ellipsoid().e2();

    // clang-format off

    apa[0] = ellipsoid().e2() * ((1. / 3.) + (ellipsoid().e2() * ((31. / 180.) +
            (ellipsoid().e2() * (517. / 5040.)))));
    apa[1] = std::pow(ellipsoid().e2(), 2) *
             ((23. / 360.) + (ellipsoid().e2() * (251. / 3780.)));
    apa[2] = std::pow(ellipsoid().e2(), 3) * (761. / 45360.);
    qp = pj_qsfn(1., e, one_es);

    // clang-format on
}

int CEA::forward(const Vec3& llh, Vec3& enu) const
{
    enu[0] = k0 * llh[0] * ellipsoid().a();
    enu[1] = (.5 * ellipsoid().a() * pj_qsfn(std::sin(llh[1]), e, one_es)) / k0;
    enu[2] = llh[2];
    return 0;
}

int CEA::inverse(const Vec3& enu, Vec3& llh) const
{
    llh[0] = enu[0] / (k0 * ellipsoid().a());
    double beta = std::asin((2. * enu[1] * k0) / (ellipsoid().a() * qp));
    llh[1] = beta + (apa[0] * std::sin(2. * beta)) +
             (apa[1] * std::sin(4. * beta)) + (apa[2] * std::sin(6. * beta));
    llh[2] = enu[2];
    return 0;
}

ProjectionBase* createProj(int epsgcode)
{
    // Check for Lat/Lon
    if (epsgcode == 4326) {
        return new LonLat;
    }
    // Check for Geocentric
    else if (epsgcode == 4978) {
        return new Geocent;
    }
    // Check if UTM
    else if (epsgcode > 32600 && epsgcode < 32800) {
        return new UTM {epsgcode};
    }
    // Check if Polar Stereo
    else if (epsgcode == 3031 || epsgcode == 3413) {
        return new PolarStereo {epsgcode};
    }
    // EASE2 grid
    else if (epsgcode == 6933) {
        return new CEA;
    } else {
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                                          "Unknown EPSG code (in factory): " +
                                                  std::to_string(epsgcode));
    }
}

int projTransform(ProjectionBase* in, ProjectionBase* out, const Vec3& inpts,
                  Vec3& outpts)
{
    if (in->code() == out->code()) {
        // If input/output projections are the same don't even bother processing
        outpts = inpts;
        return 0;
    } else {
        Vec3 temp;
        if (in->inverse(inpts, temp) != 0)
            return -2;
        if (out->forward(temp, outpts) != 0)
            return 2;
    }
    return 0;
}

}} // namespace isce3::core
