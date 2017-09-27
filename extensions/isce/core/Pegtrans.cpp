//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "isce/core/Constants.h"
#include "isce/core/Ellipsoid.h"
#include "isce/core/LinAlg.h"
#include "isce/core/Peg.h"
#include "isce/core/Pegtrans.h"
using isce::core::Ellipsoid;
using isce::core::LinAlg;
using isce::core::orbitConvMethod;
using isce::core::Peg;
using isce::core::Pegtrans;
using std::invalid_argument;
using std::string;
using std::to_string;
using std::vector;


void Pegtrans::radarToXYZ(Ellipsoid &elp, Peg &peg) {
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

    vector<double> llh = {peg.lat, peg.lon, 0.};
    vector<double> p(3);
    elp.latLonToXyz(p, llh);
    vector<double> up = {cos(peg.lat) * cos(peg.lon), cos(peg.lat) * sin(peg.lon), sin(peg.lat)};
    LinAlg::linComb(1., p, -radcur, up, ov);
}

void Pegtrans::convertSCHtoXYZ(vector<double> &schv, vector<double> &xyzv, orbitConvMethod ctype) {
    /*
     * Applies the affine matrix provided to convert from the radar sch coordinates to WGS-84 xyz 
     * coordinates or vice-versa
    */
   
    // Error checking
    checkVecLen(schv,3);
    checkVecLen(xyzv,3);

    vector<double> schvt(3), llh(3);
    Ellipsoid sph(radcur,0.);

    if (ctype == SCH_2_XYZ) {
        llh = {schv[1]/radcur, schv[0]/radcur, schv[2]};
        sph.latLonToXyz(schvt, llh);
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

void Pegtrans::convertSCHdotToXYZdot(vector<double> &sch, vector<double> &xyz, 
                                     vector<double> &schdot, vector<double> &xyzdot, 
                                     orbitConvMethod ctype) {
    /*
     * Applies the affine matrix provided to convert from the radar sch velociy to WGS-84 xyz 
     * velocity or vice-versa
    */
    
    checkVecLen(sch,3);
    checkVecLen(xyz,3);
    checkVecLen(schdot,3);
    checkVecLen(xyzdot,3);

    vector<vector<double>> schxyzmat(3, vector<double>(3)), xyzschmat(3, vector<double>(3));
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

void Pegtrans::SCHbasis(vector<double> &sch, vector<vector<double>> &xyzschmat, 
                        vector<vector<double>> &schxyzmat) {
    /*
     * Computes the transformation matrix from xyz to a local sch frame
     */
   
    checkVecLen(sch,3);
    check2dVecLen(xyzschmat,3,3);
    check2dVecLen(schxyzmat,3,3);

    vector<vector<double>> matschxyzp = {{-sin(sch[0]/radcur), 
                                          -(sin(sch[1]/radcur) * cos(sch[0]/radcur)), 
                                          cos(sch[0]/radcur) * cos(sch[1]/radcur)},
                                         {cos(sch[0]/radcur),  
                                          -(sin(sch[1]/radcur) * sin(sch[0]/radcur)), 
                                          sin(sch[0]/radcur) * cos(sch[1]/radcur)},
                                         {0.,                  
                                          cos(sch[1]/radcur),                         
                                          sin(sch[1]/radcur)}};
    LinAlg::matMat(mat, matschxyzp, schxyzmat);
    LinAlg::tranMat(schxyzmat, xyzschmat);
}

