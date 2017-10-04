//
// Source Author: Piyush Agram
// Co-Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include "isce/core/Projections.h"
using isce::core::CEA;
using isce::core::Geocent;
using isce::core::PolarStereo;
using isce::core::ProjectionBase;
using isce::core::UTM;
using std::cout;
using std::endl;
using std::invalid_argument;
using std::string;
using std::to_string;
using std::vector;

/* * * * * * * * * * * * * * * * * * * * Geocent Projection * * * * * * * * * * * * * * * * * * * */
int Geocent::forward(const vector<double> &llh, vector<double>& xyz) const {
    /*
     * This is to transform LLH to Geocent, which is just a pass-through to latLonToXyz.
     */

    // May need to implement to temporarily swap lon/lat/height to lat/lon/height for pass-through
    // vector<double> llh_swapped = {llh[1], llh[0], llh[2]};
    // ellipse.latLonToXyz(llh_swapped, out);
    ellipse.latLonToXyz(llh, xyz);
    return 0;
}

int Geocent::inverse(const vector<double> &xyz, vector<double>& llh) const {
    /*
     * This is to transform Geocent to LLH, which is just a pass-through to xyzToLatLon.
     */

    // May need to implement to temporarily swap lat/lon/height to lon/lat/height for output
    // ellipse.xyzToLatLon(xyz, llh);
    // llh = { llh[1], llh[0], llh[2] };
    ellipse.xyzToLatLon(xyz, llh);
    return 0;
}
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* * * * * * * * * * * * * * * * * * * * * UTM Projection * * * * * * * * * * * * * * * * * * * * */
/*  
 *  PROPOSE REMOVING - gatg() has a direct relation to clens() as clens(a, len(a), 2*B) + B == 
 *                     gatg(a, len(a), B). The internal function implementations differed only in
 *                     variable names and orders of operation that were transitive in nature. The
 *                     clens() function has been modified to generalize for both the original 
 *                     implementation as well as to accomodate for encapsulating gatg().
 *
double gatg(const double *p1, int len_p1, double B) {
    //
    // Local function - Compute a Gaussian latitude.
    //
    double const *p;
    double h = 0.0;
    double h2 = 0.0;
    double cos_2B;

    cos_2B = 2 * cos(2 * B);
    for(double const *p = p1 + 6, double h1 = *(--p); p-p1; h2 = h1, h1 = h)
        h = -h2 + cos_2B*h1 + *--p;

    return (B + h * sin(2*B));
}
*/
double clens(const double *a, int size, double real) {
    /*
     * Local function - Compute the real clenshaw summation. Also computes Gaussian latitude for
     * some B as (clens(a, len(a), 2*B) + B).
     *
     * NOTE: The implementation here has been modified to allow for encapsulating the gatg()
     *       implementation, as well as to make the implementation details much clearer/cleaner.
     */
    const double *p;
    double hr, hr1, hr2;
    for (p = a + size, hr2 = 0., hr1 = *(--p), hr=0.; 
         a - p > 0; 
         hr2 = hr1, hr1 = hr) {
        hr = -hr2 + (2. * hr1 * cos(real)) + *(--p);
    }
    return sin(real) * hr;    
}

double clenS(const double *a, int size, double real, double imag, double &R, double &I) {
    /*
     * Local function - Compute the complex clenshaw summation.
     *
     * NOTE: The implementation here has been modified to match the modified implementation of the
     *       real clenshaw summation above. As expected with complex->real behavior, if imag == 0,
     *       then I == 0 on return regardless of other inputs (so maybe we just implement
     *       clenS(a,len(a),real,0,_,_) for clens(a,len(a),real) to simplify the code space?)
     */
    const double *p;
    double hr, hr1, hr2, hi, hi1, hi2;
    for (p = a + size, hr2 = 0., hi2 = 0., hi1 = 0., hr1 = *(--p), hi1 = 0., hr = 0., hi = 0.;
         a - p > 0;
         hr2 = hr1, hi2 = hi1, hr1 = hr, hi1 = hi) {
        hr = -hr2 + (2. * hr1 * cos(real) * cosh(imag)) - (-2. * hi1 * sin(real) * sinh(imag)) + 
             *(--p);
        hi = -hi2 + (-2. * hr1 * sin(real) * sinh(imag)) + (2. * hi1 * cos(real) * cosh(imag));
    }
    // Bad practice - Should *either* modify R in-place *or* return R, not both. I is modified, but
    // not returned. Since R and I are tied, we should either return a pair<,>(,) or modify
    // in-place, not mix the strategies
    R = (sin(real) * cosh(imag) * hr) - (cos(real) * sinh(imag) * hi);
    I = (sin(real) * cosh(imag) * hi) + (cos(real) * sinh(imag) * hr);
    return R;
}

UTM::UTM(int code) : ProjectionBase(code) {
    /*
     * Value constructor, delegates to base constructor before continuing with UTM-specific setup
     * code (previously contained in a private _setup() method but moved given that _setup() was
     * not supposed to be callable after construction).
     */
    if ((_epsgcode > 32600) && (_epsgcode <= 32660)) {
        zone = _epsgcode - 32600;
        isnorth = true;
    } else if ((_epsgcode > 32700) && (_epsgcode <= 32760)) {
        zone = _epsgcode - 32700;
        isnorth = false;
    } else {
        string errstr = "In UTM::UTM - Invalid EPSG Code for UTM Projection. Received ";
        errstr += to_string(_epsgcode);
        errstr += ", expected in ranges (32600,32660] or (32700,32760].";
        throw invalid_argument(errstr);
    }

    lon0 = ((zone - 0.5) * (M_PI / 30.)) - M_PI;

    // Ellipsoid flattening
    double f = ellipse.e2 / (1. + sqrt(1 - ellipse.e2));
    // Third flattening
    double n = f / (2. - f);

    // Gaussian -> Geodetic == cgb
    // Geodetic -> Gaussian == cbg
    cgb[0] = n * (2 + n * ((-2./3.) + n * (-2 + n * ((116./45.) + n * ((26./45.) + 
                                                                       n * (2854./675.))))));
    cbg[0] = n * (-2 + n * ((2./3.) + n * ((4./3.) + n * ((-82./45.) + n * ((32./45.) + 
                                                                            n * (4642./4725.))))));
    cgb[1] = pow(n,2) * ((7./3.) + n * ((-8./5.) + n * ((-227./45.) + n * ((2704./315.) + 
                                                                     n * (2323./945.)))));
    cbg[1] = pow(n,2) * ((5./3.) + n * ((-16./15.) + n * ((-13./9.) + n * ((904./315.) + 
                                                                     n * (-1522./945.)))));
    cgb[2] = pow(n,3) * ((56./15.) + n * ((-136./35.) + n * ((-1262./105.) + n * (73814./2835.))));
    cbg[2] = pow(n,3) * ((-26./15.) + n * ((34./21.) + n * ((8./5.) + n * (-12686./2835.))));
    cgb[3] = pow(n,4) * ((4279./630.) + n * ((-332./35.) + n * (-399572/14175.)));
    cbg[3] = pow(n,4) * ((1237./630.) + n * ((-12./5.) + n * (-24832./14175.)));
    cgb[4] = pow(n,5) * ((4174./315.) + n * (-144838./6237.));
    cbg[4] = pow(n,5) * ((-734./315.) + n * (109598./31185.));
    cgb[5] = pow(n,6) * (601676./22275.);
    cbg[5] = pow(n,6) * (444337./155925.);

    // We have fixed k0 = 0.9996 here. This is standard for WGS84 zones. Proj4 allows this to be
    // changed for custom definitions. We plan to support standard definitions only.
    Qn = (0.9996 / (1. + n)) * (1. + n * n * ((1./4.) + n * n * ((1./64.) + ((n * n) / 256.))));

    // Elliptical N,E -> Spherical N,E == utg
    // Spherical N,E -> Elliptical N,E == gtu
    utg[0] = n * (-.5 + n * ((2./3.) + n * ((-37./96.) + n * ((1./360.) +
                                                              n * ((81./512.) +
                                                                   n * (-96199./604800.))))));
    gtu[0] = n * (.5 + n * ((-2./3.) + n * ((5./16.) + n * ((41./180.) +
                                                            n * ((-127./288.) +
                                                                 n * (7891./37800.))))));
    utg[1] = pow(n,2) * ((-1./48.) + n * ((-1./15.) + n * ((437./1440.) + 
                                                           n * ((-46./105.) + 
                                                                n * (1118711./3870720.)))));
    gtu[1] = pow(n,2) * ((13./48.) + n * ((-3./5.) + n * ((557./1440.) +
                                                          n * ((281./630.) + 
                                                               n * (-1983433./1935360.)))));
    utg[2] = pow(n,3) * ((-17./480.) + n * ((37./840.) + n * ((209./4480.) +
                                                              n * (-5569./90720.))));
    gtu[2] = pow(n,3) * ((61./240.) + n * ((-103./140.) + n * ((15061./26880.) +
                                                               n * (167603./181440.))));
    utg[3] = pow(n,4) * ((-4397./161280.) + n * ((11./504.) + n * (830251./7257600.)));
    gtu[3] = pow(n,4) * ((49561./161280.) + n * ((-179./168.) + n * (6601661./7257600.)));
    utg[4] = pow(n,5) * ((-4583./161280.) + n * (108847./3991680.));
    gtu[4] = pow(n,5) * ((34729./80640.) + n * (-3418889./1995840.));
    utg[5] = pow(n,6) * (-20648693./638668800.);
    gtu[5] = pow(n,6) * (212378941./319334400.);

    // Gaussian latitude of origin latitude
    //double Z = gatg(cbg, 6, 0.);
    double Z = clens(cbg, 6, 0.); // clens(a,len(a),2*b) + b == gatg(a,len(a),b)
    Zb = -Qn * (Z + clens(gtu, 6, 2*Z));
}

int UTM::forward(const vector<double> &llh, vector<double> &utm) const {
    /*
     * Transform from LLH to UTM.
     */
    // Elliptical Lat, Lon -> Gaussian Lat, Lon
    //double Cn = gatg(cbg, 6, llh[1]);
    double Cn = clens(cbg, 6, 2.*llh[1]) + llh[1]; // clens(a,len(a),2*B) + B == gatg(a,len(a),B)
    // Adjust longitude for zone offset
    double lam = llh[0] - lon0;

    // Account for longitude and get Spherical N,E
    Cn = atan2(sin(Cn), cos(lam)*cos(Cn));
    double Ce = atan2(sin(lam)*cos(Cn), hypot(sin(Cn), cos(Cn)*cos(lam)));

    //Spherical N,E to Elliptical N,E
    Ce = asinh(tan(Ce));
    double dCn, dCe;
    Cn += clenS(gtu, 6, 2*Cn, 2*Ce, dCn, dCe);
    Ce += dCe;


    if (abs(Ce) <= 2.623395162778) {
        utm[0] = (Qn * Ce * ellipse.a) + 500000.;
        utm[1] = (((Qn * Cn) + Zb) * ellipse.a) + (isnorth ? 0. : 10000000.);
        // UTM is lateral projection only, height is pass through.
        utm[2] = llh[2];
        return 0;
    } else {
        return 1;
    }
}

int UTM::inverse(const vector<double> &utm, vector<double> &llh) const {
    /*
     * Transform from UTM to LLH.
     */
    double Cn = (utm[1] - (isnorth ? 0. : 10000000.)) / ellipse.a;
    double Ce = (utm[0] - 500000.) / ellipse.a;

    //Normalize N,E to Spherical N,E
    Cn = (Cn - Zb) / Qn;
    Ce /= Qn;

    if (abs(Ce) <= 2.623395162778) {
        //N,E to Spherical Lat, Lon
        double dCn, dCe;
        Cn += clenS(utg, 6, 2*Cn, 2*Ce, dCn, dCe);
        Ce = atan(sinh(Ce + dCe));
        
        //Spherical Lat, Lon to Gaussian Lat, Lon
        Ce = atan2(sin(Ce), cos(Ce)*cos(Cn));
        Cn = atan2(sin(Cn)*cos(Ce), hypot(sin(Ce), cos(Ce)*cos(Cn)));

        //Gaussian Lat, Lon to Elliptical Lat, Lon
        llh[0] = Ce + lon0;
        //llh[1] = gatg(cgb, 6, Cn);
        llh[1] = clens(cgb, 6, 2*Cn) + Cn; // clens(a,len(a),2*B) + B == gatg(a,len(a),B)
        //UTM is a lateral projection only. Height is pass through.
        llh[2] = utm[2];
        return 0;
    } else {
        return 1;
    }
}
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* * * * * * * * * * * * * * * * * * * PolarStereo Projection * * * * * * * * * * * * * * * * * * */
double pj_tsfn(double phi, double sinphi, double e) {
    /*
     * Local function - Determine small t from PROJ.4.
     */
    sinphi *= e;
    return tan(.5 * ((.5*M_PI) - phi)) / pow((1. - sinphi) / (1. + sinphi), .5*e);
}

PolarStereo::PolarStereo(int code) : ProjectionBase(code) {
    /*
     * Set up various parameters for polar stereographic projection. Currently only EPSG:3031
     * (Antarctic) and EPSG:3413 (Greenland) are supported.
     */
    if (_epsgcode == 3031) {
        isnorth = false;
        lat0 = -M_PI / 2.;
        // Only need absolute value
        lat_ts = (71. * M_PI) / 180.;
        lon0 = 0.;
    } else if (_epsgcode == 3413) {
        isnorth = true;
        lat0 = M_PI / 2.;
        lat_ts = 70. * (M_PI / 180.);
        lon0 = -45. * (M_PI / 180.);
    } else {
        string errstr = "In PolarStereo::PolarStereo - Invalid EPSG Code for Polar Stereographic ";
        errstr += "projection. Received ";
        errstr += to_string(_epsgcode);
        errstr += ", expected either 3031 (Antarctic) or 3413 (Greenland). [NOTE: Other codes are ";
        errstr += "currently not supported]";
        throw invalid_argument(errstr);
    }
    e = sqrt(ellipse.e2);
    akm1 = cos(lat_ts) / pj_tsfn(lat_ts, sin(lat_ts), e);
    akm1 *= ellipse.a / sqrt(1. - (pow(e,2) * pow(sin(lat_ts),2)));
}

int PolarStereo::forward(const vector<double> &llh, vector<double> &out) const {
    /*
     * Transform from LLH to Polar Stereo.
     */
    double lam = llh[0] - lon0;
    double phi = llh[1] * (isnorth ? 1. : -1.);
    double temp = akm1 * pj_tsfn(phi, sin(phi), e);

    out[0] = temp * sin(lam);
    out[1] = -temp * cos(lam) * (isnorth ? 1. : -1.);
    //Height is just pass through
    out[2] = llh[2];

    return 0;
}

int PolarStereo::inverse(const vector<double> &ups, vector<double> &llh) const {
    /*
     * Transform from Polar Stereo to LLH.
     */
    double tp = -hypot(ups[0], ups[1]) / akm1;
    double phi_l = (.5*M_PI) - (2. * atan(tp));

    double sinphi;
    double phi = 0.;
    for(int i=8; i--; phi_l = phi) {
        sinphi = e * sin(phi);
        phi = 2. * atan((tp * pow((1. + sinphi) / (1. - sinphi), -.5 * e)) + (.5*M_PI));
        if (abs(phi_l - phi) < 1.e-10) {
            if (isnorth) {
                llh[0] = ((ups[0] == 0.) && (ups[1] == 0.)) ? 0. : atan2(ups[0], -ups[1]);
                llh[1] = phi;
                llh[2] = ups[2];
                return 0;
            } else {
                llh[0] = ((ups[0] == 0.) && (ups[1] == 0.)) ? 0. : atan2(ups[0], ups[1]);
                llh[1] = -phi;
                llh[2] = ups[2];
                return 0;
            }
        }
    }
    return 1;
}
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* * * * * * * * * * * * * * * * * * * * * CEA Projection * * * * * * * * * * * * * * * * * * * * */
double pj_qsfn(double sinphi, double e, double one_es) {
    /*
     * Local function - ???
     */
    double con = e * sinphi;
    return one_es * ((sinphi / (1. - pow(con,2))) - ((.5 / e) * log((1. - con) / (1. + con))));
}

CEA::CEA() : ProjectionBase(6933) {
    /*
     * Set up parameters for equal area projection.
     */
    lat_ts = M_PI / 6.;
    k0 = cos(lat_ts) / sqrt(1. - (ellipse.e2 * pow(sin(lat_ts),2)));
    e = sqrt(ellipse.e2);
    one_es = 1. - ellipse.e2;
    apa[0] = ellipse.e2 * ((1./3.) + (ellipse.e2 * ((31./180.) + (ellipse.e2 * (517./5040.)))));
    apa[1] = pow(ellipse.e2,2) * ((23./360.) + (ellipse.e2 * (251./3780.)));
    apa[2] = pow(ellipse.e2,3) * (761./45360.);
    qp = pj_qsfn(1., e, one_es);
}

int CEA::forward(const vector<double> &llh, vector<double> &enu) const {
    /*
     * Transform from LLH to CEA.
     */
    enu[0] = k0 * llh[0] * ellipse.a;
    enu[1] = (.5 * ellipse.a * pj_qsfn(sin(llh[1]), e, one_es)) / k0;
    enu[2] = llh[2];
    return 0;
}

int CEA::inverse(const vector<double> &enu, vector<double> &llh) const {
    /*
     * Transform from LLH to CEA.
     */
    llh[0] = enu[0] / (k0 * ellipse.a);
    double beta = asin((2. * enu[1] * k0) / (ellipse.a * qp));
    llh[1] = beta + (apa[0] * sin(2. * beta)) + (apa[1] * sin(4. * beta)) + 
             (apa[2] * sin(6. * beta));  
    llh[2] = enu[2];
    return 0;
}
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* * * * * * * * * * * * * * * * * * * Projection Transformer * * * * * * * * * * * * * * * * * * */
int projTransform(ProjectionBase &in, ProjectionBase &out, const vector<double> &inpts, 
                  vector<double> &outpts) {
    if (in._epsgcode == out._epsgcode) {
        // If input/output projections are the same don't even bother processing
        outpts = inpts;
        return 0;
    } else if (in._epsgcode == 4326) {
        // Consider case where input is Lat/Lon
        return out.forward(inpts, outpts);
    } else if (out._epsgcode == 4326) {
        // Consider case where output is Lat/Lon
        return -out.inverse(inpts, outpts);
    } else {
        vector<double> temp(3);
        if (in.inverse(inpts, temp) != 0) return -2;
        if (out.forward(temp, outpts) != 0) return 2;
    }
    return 0;
};
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

