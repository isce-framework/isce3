/**
 * Source Author: Paulo Penteado, based on Projections.cpp by Piyush Agram / Joshua Cohen
 * Copyright 2018
 */

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include "gpuProjections.h"
#include <isce/cuda/except/Error.h>
using isce::core::Vec3;

namespace isce { namespace cuda { namespace core {

//Helper for the host side function - used only for testing
__global__ void forward_g(int code,
                          ProjectionBase** base,
                          const double *inpts,
                          double *outpts,
                          int *flags)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*base) = createProj(code);
        flags[0] = (*base)->forward(*(Vec3*) inpts, *(Vec3*) outpts);
        delete *base;
    }
}

//Helper for the host side function - used only for testing
__global__ void inverse_g(int code,
                          ProjectionBase **base,
                          const double *inpts,
                          double *outpts,
                          int *flags)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*base) = createProj(code);
        flags[0] = (*base)->inverse(*(Vec3*) inpts, *(Vec3*) outpts);
        delete *base;
    }
}

__host__ int ProjectionBase::forward_h(const Vec3& llh, Vec3& xyz) const
{
    /*
     * This is to transfrom from LLH to requested projection system on the host.
     */

    double *llh_d, *xyz_d;
    int *flag_d;
    ProjectionBase **base_d;

    checkCudaErrors( cudaMalloc((int**)&flag_d, 1*sizeof(int)));
    checkCudaErrors( cudaMalloc((double**)&llh_d,3*sizeof(double)));
    checkCudaErrors( cudaMalloc((double**)&xyz_d,3*sizeof(double)));
    checkCudaErrors( cudaMalloc(&base_d, sizeof(ProjectionBase**)));
    checkCudaErrors( cudaMemcpy(llh_d, llh.data(), 3*sizeof(double), cudaMemcpyHostToDevice));

    //Call the global function with a single thread
    forward_g<<<1,1>>>(_epsgcode, base_d, llh_d, xyz_d, flag_d);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors( cudaMemcpy(xyz.data(), xyz_d, 3*sizeof(double), cudaMemcpyDeviceToHost));
    int status;
    checkCudaErrors( cudaMemcpy(&status, flag_d, sizeof(int), cudaMemcpyDeviceToHost));

    //Clean up
    checkCudaErrors( cudaFree(llh_d));
    checkCudaErrors( cudaFree(xyz_d));
    checkCudaErrors( cudaFree(flag_d));
    checkCudaErrors( cudaFree(base_d));
    return status;
}

__host__ int ProjectionBase::inverse_h(const Vec3& xyz, Vec3& llh) const
{
    /*
     * This is to transfrom from requested projection system to LLH on the host.
     */
    double *llh_d, *xyz_d;
    int *flag_d;
    ProjectionBase **base_d;
    checkCudaErrors( cudaMalloc((int**)&flag_d, sizeof(int)));
    checkCudaErrors( cudaMalloc((double**)&llh_d,3*sizeof(double)));
    checkCudaErrors( cudaMalloc((double**)&xyz_d,3*sizeof(double)));
    checkCudaErrors( cudaMalloc(&base_d, sizeof(ProjectionBase**)));
    checkCudaErrors( cudaMemcpy(xyz_d, xyz.data(), 3*sizeof(double), cudaMemcpyHostToDevice));

    //Call the global function with a single thread
    inverse_g<<<1,1>>>(_epsgcode*1, base_d, xyz_d, llh_d, flag_d);
    checkCudaErrors( cudaDeviceSynchronize());
    checkCudaErrors( cudaMemcpy(llh.data(), llh_d, 3*sizeof(double), cudaMemcpyDeviceToHost));
    int status;
    checkCudaErrors( cudaMemcpy(&status, flag_d, sizeof(int), cudaMemcpyDeviceToHost));

    //Clean up
    checkCudaErrors( cudaFree(llh_d));
    checkCudaErrors( cudaFree(xyz_d));
    checkCudaErrors( cudaFree(flag_d));
    checkCudaErrors( cudaFree(base_d));
    return status;
}

CUDA_DEV int LonLat::forward(const Vec3& in, Vec3& out) const
{
    /*
     * Transforms Lon/Lat from radians to degrees.
     */
    out[0] = in[0] * 180.0/M_PI;
    out[1] = in[1] * 180.0/M_PI;
    out[2] = in[2];
    return 0;
}

CUDA_DEV int LonLat::inverse(const Vec3& in, Vec3& out) const
{
    /*
     * Transforms Lon/Lat from degrees to radians.
     */
    out[0] = in[0] * M_PI/180.0;
    out[1] = in[1] * M_PI/180.0;
    out[2] = in[2];
    return 0;
}

CUDA_DEV int Geocent::forward(const Vec3& in, Vec3& out) const
{
    /*
     * Same as Ellipsoid::lonLatToXyz.
     */
    ellipse.lonLatToXyz(in, out);
    return 0;
}

CUDA_DEV int Geocent::inverse(const Vec3& in, Vec3& out) const
{
    /*
     * Same as Ellipsoid::xyzToLonLat
     */
    ellipse.xyzToLonLat(in, out);
    return 0;
}

CUDA_HOSTDEV double clens(const double* a, int size, double real) {
    /*
     * Local function - Compute the real clenshaw summation. Also computes Gaussian latitude for
     * some B as clens(a, len(a), 2*B) + B.
     *
     * NOTE: The implementation here has been modified to allow for encapsulating the gatg()
     *       implementation, as well as to make the implementation details much clearer/cleaner.
     */
    const double *p;
    double hr, hr1, hr2;
    for (p = a + size, hr2 = 0., hr1 = *(--p), hr=0.;
         a - p;
         hr2 = hr1, hr1 = hr) {
        hr = -hr2 + (2. * hr1 * cos(real)) + *(--p);
    }
    return sin(real) * hr;
}


CUDA_HOSTDEV double clenS(const double *a, int size, double real, double imag, double &R, double &I) {
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
         a - p;
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


CUDA_HOSTDEV UTM::UTM(int code) : ProjectionBase(code) {
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
	//Error handling delegated to CPU side
    }

    lon0 = ((zone - 0.5) * (M_PI / 30.)) - M_PI;

    // Ellipsoid flattening
    double f = ellipse.e2() / (1. + sqrt(1 - ellipse.e2()));
    // Third flattening
    double n = f / (2. - f);

    // Gaussian -> Geodetic == cgb
    // Geodetic -> Gaussian == cbg
    cgb[0] = n * (2 + n * ((-2./3.) + n * (-2 + n * ((116./45.) + n * ((26./45.) + n * (-2854./675.))))));
    cbg[0] = n * (-2 + n * ((2./3.) + n * ((4./3.) + n * ((-82./45.) + n * ((32./45.) + n * (4642./4725.))))));
    cgb[1] = pow(n,2) * ((7./3.) + n * ((-8./5.) + n * ((-227./45.) + n * ((2704./315.) + n * (2323./945.)))));
    cbg[1] = pow(n,2) * ((5./3.) + n * ((-16./15.) + n * ((-13./9.) + n * ((904./315.) + n * (-1522./945.)))));
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
    // JC - clens(_,_,0.) is always 0, should we hardcode/eliminate this?
    double Z = clens(cbg, 6, 0.);
    Zb = -Qn * (Z + clens(gtu, 6, 2*Z));
}

CUDA_DEV int UTM::forward(const Vec3& llh, Vec3& utm) const {
    /*
     * Transform from LLH to UTM.
     */
    // Elliptical Lat, Lon -> Gaussian Lat, Lon
    double gauss = clens(cbg, 6, 2.*llh[1]) + llh[1];
    // Adjust longitude for zone offset
    double lam = llh[0] - lon0;

    // Account for longitude and get Spherical N,E
    double Cn = atan2(sin(gauss), cos(lam)*cos(gauss));
    double Ce = atan2(sin(lam)*cos(gauss), hypot(sin(gauss), cos(gauss)*cos(lam)));

    //Spherical N,E to Elliptical N,E
    Ce = asinh(tan(Ce));
    double dCn, dCe;
    Cn += clenS(gtu, 6, 2*Cn, 2*Ce, dCn, dCe);
    Ce += dCe;

    if (fabs(Ce) <= 2.623395162778) {
        utm[0] = (Qn * Ce * ellipse.a()) + 500000.;
        utm[1] = (((Qn * Cn) + Zb) * ellipse.a()) + (isnorth ? 0. : 10000000.);
        // UTM is lateral projection only, height is pass through.
        utm[2] = llh[2];
        return 0;
    } else {
        return 1;
    }
}


CUDA_DEV int UTM::inverse(const Vec3& utm, Vec3& llh) const {
    /*
     * Transform from UTM to LLH.
     */
    double Cn = (utm[1] - (isnorth ? 0. : 10000000.)) /  ellipse.a();
    double Ce = (utm[0] - 500000.) /  ellipse.a();

    //Normalize N,E to Spherical N,E
    Cn = (Cn - Zb) / Qn;
    Ce /= Qn;

    if (fabs(Ce) <= 2.623395162778) {
        //N,E to Spherical Lat, Lon
        double dCn, dCe;
        Cn += clenS(utg, 6, 2*Cn, 2*Ce, dCn, dCe);
        Ce = atan(sinh(Ce + dCe));

        //Spherical Lat, Lon to Gaussian Lat, Lon
        double sinCe = sin(Ce);
        double cosCe = cos(Ce);
        Ce = atan2(sinCe, cosCe*cos(Cn));
        Cn = atan2(sin(Cn)*cosCe, hypot(sinCe, cosCe*cos(Cn)));

        //Gaussian Lat, Lon to Elliptical Lat, Lon
        llh[0] = Ce + lon0;
        llh[1] = clens(cgb, 6, 2*Cn) + Cn;
        //UTM is a lateral projection only. Height is pass through.
        llh[2] = utm[2];
        return 0;
    } else {
        return 1;
    }
} 

CUDA_HOSTDEV double pj_tsfn(double phi, double sinphi, double e) {
    /*
     * Local function - Determine small t from PROJ.4.
     */
    sinphi *= e;
    return tan(.5 * ((.5*M_PI) - phi)) / pow((1. - sinphi) / (1. + sinphi), .5*e);
}

CUDA_HOSTDEV PolarStereo::PolarStereo(int code) : ProjectionBase(code) {
    /*
     * Set up various parameters for polar stereographic projection. Currently only EPSG:3031
     * (Antarctic) and EPSG:3413 (Greenland) are supported.
     */
    if (_epsgcode == 3031) {
        isnorth = false;
        // Only need absolute value
        lat_ts = (71. * M_PI) / 180.;
        lon0 = 0.;
    } else if (_epsgcode == 3413) {
        isnorth = true;
        lat_ts = 70. * (M_PI / 180.);
        lon0 = -45. * (M_PI / 180.);
    } else {
            //Need to figure out a way to throw error on device
            //Currently, delegated to CPU side
    }
    e = sqrt(ellipse.e2());
    akm1 = cos(lat_ts) / pj_tsfn(lat_ts, sin(lat_ts), e);
    akm1 *= ellipse.a() / sqrt(1. - (pow(e,2) * pow(sin(lat_ts),2)));
}

CUDA_DEV int PolarStereo::forward(const Vec3& llh, Vec3& out)  const{
    /**
     * Host / Device forward projection function.
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

CUDA_DEV int PolarStereo::inverse(const Vec3& ups, Vec3& llh) const {
    /**
    * Host / Device inverse projection function.
    */
    double tp = -hypot(ups[0], ups[1])/akm1;
    double fact = (isnorth)?1:-1;
    double phi_l = (.5*M_PI) - (2. * atan(tp));

    double sinphi;
    double phi = 0.;
    for(int i=8; i--; phi_l = phi) {
        sinphi = e * sin(phi_l);
        phi = 2. * atan(tp * pow((1. + sinphi) / (1. - sinphi), -0.5*e)) +0.5 * M_PI;
        if (fabs(phi_l - phi) < 1.e-10) {
            llh[0] = ((ups[0] == 0.) && (ups[1] == 0.)) ? 0. : atan2(ups[0], -fact*ups[1]) + lon0;
            llh[1] = phi*fact;
            llh[2] = ups[2];
            return 0;
        }
    }
    return 1;
}

CUDA_HOSTDEV double pj_qsfn(double sinphi, double e, double one_es) {
    /*
     * Local function - ???
     */
    double con = e * sinphi;
    return one_es * ((sinphi / (1. - pow(con,2))) - ((.5 / e) * log((1. - con) / (1. + con))));
}

CUDA_HOSTDEV CEA::CEA() : ProjectionBase(6933) {
    /*
     * Set up parameters for equal area projection.
     */
    lat_ts = M_PI / 6.;
    k0 = cos(lat_ts) / sqrt(1. - (ellipse.e2() * pow(sin(lat_ts),2)));
    e = sqrt(ellipse.e2());
    one_es = 1. - ellipse.e2();
    apa[0] = ellipse.e2() * ((1./3.) + (ellipse.e2() * ((31./180.) + (ellipse.e2() * (517./5040.)))));
    apa[1] = pow(ellipse.e2(),2) * ((23./360.) + (ellipse.e2() * (251./3780.)));
    apa[2] = pow(ellipse.e2(),3) * (761./45360.);
    qp = pj_qsfn(1., e, one_es);
}

CUDA_DEV int CEA::forward(const Vec3& llh, Vec3& enu) const {
    /*
     * Transform from LLH to CEA.
     */
    enu[0] = k0 * llh[0] * ellipse.a();
    enu[1] = (.5 * ellipse.a() * pj_qsfn(sin(llh[1]), e, one_es)) / k0;
    enu[2] = llh[2];
    return 0;
}

CUDA_DEV int CEA::inverse(const Vec3& enu, Vec3& llh) const {
    /*
     * Transform from LLH to CEA.
     */
    llh[0] = enu[0] / (k0 * ellipse.a());
    double beta = asin((2. * enu[1] * k0) / (ellipse.a() * qp));
    llh[1] = beta + (apa[0] * sin(2. * beta)) + (apa[1] * sin(4. * beta)) + (apa[2] * sin(6. * beta));
    llh[2] = enu[2];
    return 0;
}

CUDA_HOSTDEV ProjectionBase* createProj(int epsgcode)
{
    //Check for Lat/Lon
    if (epsgcode == 4326)
    {
        return new LonLat;
    }
    //Check for geocent
    else if (epsgcode == 4978)
    {
        return new Geocent;
    }
    //Check for UTM
    else if (epsgcode > 32600 && epsgcode < 32800)
    {
        return new UTM(epsgcode);
    }
    //Check for Polar Stereo
    else if (epsgcode == 3031 || epsgcode == 3413)
    {
        return new PolarStereo(epsgcode);
    }
    //EASE2 grid
    else if (epsgcode == 6933)
    {
        return new CEA;
    }
    else
    {
        //Somehow errors must be handled at this stage
        //Delegating to CPU code
        return NULL;
    }
}

CUDA_DEV int projTransform(const ProjectionBase *in,
                           const ProjectionBase *out,
                           const Vec3& inpts,
                           Vec3& outpts) {
    if (in->_epsgcode == out->_epsgcode) {
        // If input/output projections are the same don't even bother processing
        for (int ii=0; ii<3;ii++)
            outpts[ii] = inpts[ii];
        return 0;
    } else {
        Vec3 temp;
        if (in->inverse(inpts, temp)   != 0) return -2;
        if (out->forward(temp, outpts) != 0) return  2;
    }
    return 0;
};

__device__ int projInverse(int code, const Vec3& in, Vec3& out) {
    if (code == 4326) {
        LonLat proj;
        return proj.inverse(in, out);
    } else if (code == 3031 or code == 3413) {
        PolarStereo proj(code);
        return proj.inverse(in, out);
    } else {
        return 1; // failure
    }
}

}}}
