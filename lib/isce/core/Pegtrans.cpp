//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <stdexcept>
#include "Constants.h"
#include "Ellipsoid.h"
#include "Peg.h"
#include "Pegtrans.h"
using isce::core::Ellipsoid;
using isce::core::Peg;
using isce::core::Pegtrans;
using std::invalid_argument;


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

    matinv = mat.transpose();

    radcur = elp.rDir(peg.hdg, peg.lat);

    cartesian_t llh = {peg.lon, peg.lat, 0.};
    cartesian_t p;
    elp.lonLatToXyz(llh, p);
    cartesian_t up = {cos(peg.lat) * cos(peg.lon), cos(peg.lat) * sin(peg.lon), sin(peg.lat)};
    ov = p - radcur * up;
}

void Pegtrans::convertXYZtoSCH(const cartesian_t & xyzv, cartesian_t & schv) const {
    /*
     * Applies the affine matrix provided to convert from WGS-84 xyz to radar sch
     * coordinates.
    */
    cartesian_t schvt, llh;
    // Create reference sphere
    Ellipsoid sph(radcur,0.);
    // Perform conversion
    schvt = xyzv - ov;
    schv = matinv.dot(schvt);
    sph.xyzToLonLat(schv, llh);
    schv = {radcur*llh[0], radcur*llh[1], llh[2]};
}

void Pegtrans::convertSCHtoXYZ(const cartesian_t & schv, cartesian_t & xyzv) const {
    /*
     * Applies the affine matrix provided to convert from the radar sch coordinates to WGS-84 xyz
     * coordinates.
    */
    cartesian_t schvt, llh;
    // Create reference sphere
    Ellipsoid sph(radcur,0.);
    // Perform conversion
    llh = {schv[0]/radcur, schv[1]/radcur, schv[2]};
    sph.lonLatToXyz(llh, schvt);
    xyzv = mat.dot(schvt);
    xyzv = xyzv + ov;
}

void Pegtrans::convertXYZdotToSCHdot(const cartesian_t & sch, const cartesian_t & xyzdot,
                                     cartesian_t & schdot) const {
    /*
     * Applies the affine matrix provided to convert from the WGS-84 xyz
     * velocity to radar sch velocity.
    */
    cartmat_t schxyzmat, xyzschmat;
    SCHbasis(sch, xyzschmat, schxyzmat);
    schdot = xyzschmat.dot(xyzdot);
}

void Pegtrans::convertSCHdotToXYZdot(const cartesian_t & sch, const cartesian_t & schdot,
                                     cartesian_t & xyzdot) const {
    /*
     * Applies the affine matrix provided to convert from the radar sch velociy to WGS-84 xyz
     * velocity or vice-versa
    */
    cartmat_t schxyzmat, xyzschmat;
    SCHbasis(sch, xyzschmat, schxyzmat);
    xyzdot = schxyzmat.dot(schdot);
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
    schxyzmat = mat.dot(matschxyzp).transpose();
}
