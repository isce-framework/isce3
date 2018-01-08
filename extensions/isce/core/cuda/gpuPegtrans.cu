//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include "gpuEllipsoid.h"
#include "gpuLinAlg.h"
#include "gpuPeg.h"
#include "gpuPegtrans.h"
using isce::core::cuda::gpuEllipsoid;
using isce::core::cuda::gpuLinAlg;
using isce::core::cuda::gpuPeg;
using isce::core::cuda::gpuPegtrans;

__device__ void gpuPegtrans::radar2xyz(gpuEllipsoid &elp, gpuPeg &peg) {
    mat[0][0] = cos(peg.lat) * cos(peg.lon);
    mat[0][1] = (-sin(peg.hdg) * sin(peg.lon)) - (sin(peg.lat) * cos(peg.lon) * cos(peg.hdg));
    mat[0][2] = (sin(peg.lon) * cos(peg.hdg)) - (sin(peg.lat) * cos(peg.lon) * sin(peg.hdg));
    mat[1][0] = cos(peg.lat) * sin(peg.lon);
    mat[1][1] = (cos(peg.lon) * sin(peg.hdg)) - (sin(peg.lat) * sin(peg.lon) * cos(peg.hdg));
    mat[1][2] = (-cos(peg.lon) * cos(peg.hdg)) - (sin(peg.lat) * sin(peg.lon) * sin(peg.hdg));
    mat[2][0] = sin(peg.lat);
    mat[2][1] = cos(peg.lat) * cos(peg.hdg);
    mat[2][2] = cos(peg.lat) * sin(peg.hdg);

    radcur = elp.rDir(peg.hdg, peg.lat);

    double llh[3] = {peg.lat, peg.lon, 0.};
    double temp[3];
    elp.latLonToXyz(temp,llh);

    ov[0] = temp[0] - (radcur * cos(peg.lat) * cos(peg.lon));
    ov[1] = temp[1] - (radcur * cos(peg.lat) * sin(peg.lon));
    ov[2] = temp[2] - (radcur * sin(peg.lat));
}

__device__ void gpuPegtrans::xyz2sch(double *schv, double *xyzv) {
    double schvt[3];
    gpuLinAlg::linComb(1., xyzv, -1., ov, schvt);
    schv[0] = (mat[0][0] * schvt[0]) + (mat[1][0] * schvt[1]) + (mat[2][0] * schvt[2]);
    schv[1] = (mat[0][1] * schvt[0]) + (mat[1][1] * schvt[1]) + (mat[2][1] * schvt[2]);
    schv[2] = (mat[0][2] * schvt[0]) + (mat[1][2] * schvt[1]) + (mat[2][2] * schvt[2]);
 
    double llh[3];
    gpuEllipsoid sph(radcur,0.);
    sph.xyzToLatLon(schv, llh);
    schv[0] = radcur * llh[1];
    schv[1] = radcur * llh[0];
    schv[2] = llh[2];
}
