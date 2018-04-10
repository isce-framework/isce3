//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "Constants.h"
#include "Ellipsoid.h"
#include "LinAlg.h"
#include "Peg.h"
#include "Pegtrans.h"
using isce::core::Ellipsoid;
using isce::core::LinAlg;
using isce::core::orbitConvMethod;
using isce::core::Peg;
using isce::core::Pegtrans;
using std::invalid_argument;
using std::string;
using std::to_string;
using std::vector;


void Pegtrans::radarToXYZ(const Ellipsoid &elp, const Peg &peg) {
    /*
     * Computes the transformation matrix and translation vector needed to convert
     * between radar (s,c,h) coordinates and WGS-84 (x,y,z) coordinates
    */

    mat[0][0] = cos(peg.lat) * cos(peg.lon);
    mat[0][1] = -(sin(peg.hdg) * sin(peg.lon)) - (sin(peg.lat) * cos(peg.lon) * cos(peg.hdg));
    mat[0][2] = (sin(peg.lon) * cos(peg.hdg)) - (sin(peg.lat) * cos(peg.lon) * sin(peg.hdg));
    mat[1][0] = cos(peg.lat) * sin(peg.lon);
    mat[1][1] = (cos(peg.lon) * sin(peg.hdg)) - (sin(peg.lat) * sin(peg.lon) * cos(peg.hdg));
    mat[1][2] = -(cos(peg.lon) * cos(peg.hdg)) - (sin(peg.lat) * sin(peg.lon) * sin(peg.hdg));
    mat[2][0] = sin(peg.lat);
    mat[2][1] = cos(peg.lat) * cos(peg.hdg);
    mat[2][2] = cos(peg.lat) * sin(peg.hdg);

    LinAlg::tranMat(mat, matinv);

    radcur = elp.rDir(peg.hdg, peg.lat);

    cartesian_t llh = {peg.lat, peg.lon, 0.};
    cartesian_t p;
    elp.latLonToXyz(llh, p);
    cartesian_t up = {cos(peg.lat) * cos(peg.lon), cos(peg.lat) * sin(peg.lon), sin(peg.lat)};
    LinAlg::linComb(1., p, -radcur, up, ov);
}

void Pegtrans::convertSCHtoXYZ(cartesian_t &schv, cartesian_t &xyzv, orbitConvMethod ctype)
                               const {
    /*
     * Applies the affine matrix provided to convert from the radar sch coordinates to WGS-84 xyz
     * coordinates or vice-versa
    */
    cartesian_t schvt, llh;
    Ellipsoid sph(radcur,0.);

    if (ctype == SCH_2_XYZ) {
        llh = {schv[1]/radcur, schv[0]/radcur, schv[2]};
        sph.latLonToXyz(llh, schvt);
        LinAlg::matVec(mat, schvt, xyzv);
        LinAlg::linComb(1., xyzv, 1., ov, xyzv);
    } else if (ctype == XYZ_2_SCH) {
        LinAlg::linComb(1., xyzv, -1., ov, schvt);
        LinAlg::matVec(matinv, schvt, schv);
        sph.xyzToLatLon(schv, llh);
        schv = {radcur*llh[1], radcur*llh[0], llh[2]};
    } else {
        string errstr = "Unrecognized conversion type in Pegtrans::convertSCHtoXYZ.\n";
        errstr += "Expected one of:\n";
        errstr += "  SCH_2_XYZ (== "+to_string(SCH_2_XYZ)+")\n";
        errstr += "  XYZ_2_SCH (== "+to_string(XYZ_2_SCH)+")\n";
        errstr += "Encountered conversion type "+to_string(ctype);
        throw invalid_argument(errstr);
    }
}

void Pegtrans::convertSCHdotToXYZdot(const cartesian_t &sch, const cartesian_t &xyz,
                                     cartesian_t &schdot, cartesian_t &xyzdot,
                                     orbitConvMethod ctype) const {
    /*
     * Applies the affine matrix provided to convert from the radar sch velociy to WGS-84 xyz
     * velocity or vice-versa
    */
    cartmat_t schxyzmat, xyzschmat;
    SCHbasis(sch, xyzschmat, schxyzmat);

    if (ctype == SCH_2_XYZ) LinAlg::matVec(schxyzmat, schdot, xyzdot);
    else if (ctype == XYZ_2_SCH) LinAlg::matVec(xyzschmat, xyzdot, schdot);
    else {
        string errstr = "Unrecognized conversion type in Pegtrans::convertSCHdotToXYZdot.\n";
        errstr += "Expected one of:\n";
        errstr += "  SCH_2_XYZ (== "+to_string(SCH_2_XYZ)+")\n";
        errstr += "  XYZ_2_SCH (== "+to_string(XYZ_2_SCH)+")\n";
        errstr += "Encountered conversion type "+to_string(ctype);
        throw invalid_argument(errstr);
    }
}

void Pegtrans::SCHbasis(const cartesian_t &sch, cartmat_t & xyzschmat,
                        cartmat_t & schxyzmat) const {
    /*
     * Computes the transformation matrix from xyz to a local sch frame
     */
    cartmat_t matschxyzp = {{{-sin(sch[0]/radcur),
                             -(sin(sch[1]/radcur) * cos(sch[0]/radcur)),
                             cos(sch[0]/radcur) * cos(sch[1]/radcur)},
                            {cos(sch[0]/radcur),
                             -(sin(sch[1]/radcur) * sin(sch[0]/radcur)),
                             sin(sch[0]/radcur) * cos(sch[1]/radcur)},
                            {0.,
                             cos(sch[1]/radcur),
                             sin(sch[1]/radcur)}}};
    LinAlg::matMat(mat, matschxyzp, schxyzmat);
    LinAlg::tranMat(schxyzmat, xyzschmat);
}
