//
// Source Author: Paulo Penteado, based on Projections.cpp by Piyush Agram / Joshua Cohen
// Copyright 2018
//

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include "Projections.h"
#include "gpuProjections.h"
using isce::cuda::core::PolarStereo;
using isce::cuda::core::ProjectionBase;
using isce::core::cartesian_t;
using std::cout;
using std::endl;
using std::invalid_argument;
using std::string;
using std::to_string;
using std::vector;


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* * * * * * * * * * * * * * * * * * * PolarStereo Projection * * * * * * * * * * * * * * * * * * */
namespace isce { namespace cuda { namespace core {
__host__ __device__ double pj_tsfn(double phi, double sinphi, double e) {
    /*
     * Local function - Determine small t from PROJ.4.
     */
    sinphi *= e;
    return tan(.5 * ((.5*M_PI) - phi)) / pow((1. - sinphi) / (1. + sinphi), .5*e);
}
}}}




   __host__ __device__ PolarStereo::PolarStereo(int code) : ProjectionBase(code) {
    /*
     * Set up various parameters for polar stereographic projection. Currently only EPSG:3031
     * (Antarctic) and EPSG:3413 (Greenland) are supported.
     */
    if (_epsgcode == 3031) {
        isnorth = false;
        //lat0 = -M_PI / 2.;
        // Only need absolute value
        lat_ts = (71. * M_PI) / 180.;
        lon0 = 0.;
    } else if (_epsgcode == 3413) {
        isnorth = true;
        //lat0 = M_PI / 2.;e
        lat_ts = 70. * (M_PI / 180.);
        lon0 = -45. * (M_PI / 180.);
    } else {
//        string errstr = "In PolarStereo::PolarStereo - Invalid EPSG Code for Polar Stereographic ";
//        errstr += "projection. Received ";
//        errstr += to_string(_epsgcode);
//        errstr += ", expected either 3031 (Antarctic) or 3413 (Greenland). [NOTE: Other codes are ";
//        errstr += "currently not supported]";
        //throw invalid_argument(errstr);
    }
    e = sqrt(ellipse.gete2());
    akm1 = cos(lat_ts) / pj_tsfn(lat_ts, sin(lat_ts), e);
    akm1 *= ellipse.geta() / sqrt(1. - (pow(e,2) * pow(sin(lat_ts),2)));
}




//__global__ void PolarStero::forward_g( cartesian_t llh, cartesian_t out)
//{
//    /*
//     * Just a wrapper to call the device forward function from host
//     */
//     forward(llh,out);
//  
//}
   
   __host__ int PolarStereo::forward(const cartesian_t &llh, cartesian_t &out)  const{
           

       /*
        * Host forward projection function
        */
	   
   	   //cpuproj.forward(llh, out);
   	
//       /*
//        * Transform from LLH to Polar Stereo.
//        * CUDA device function
//        */
//       double lam = llh[0] - lon0;
//       double phi = llh[1] * (isnorth ? 1. : -1.);
//       double temp = akm1 * pj_tsfn(phi, sin(phi), e);
//
//       out[0] = temp * sin(lam);
//       out[1] = -temp * cos(lam) * (isnorth ? 1. : -1.);
//       //Height is just pass through
//       out[2] = llh[2];
       	return 0;
   	
   }
   
__device__ int PolarStereo::forward(double llh[], double out[])  const{
        

    /*
     * Device forward projection function
     */	
	
	
	
    /*
     * Transform from LLH to Polar Stereo.
     * CUDA device function
     * TODO: organize shareable variables
     * TODO: test for numerical exceptions and out of bounds coordinates
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

int PolarStereo::forward_h_single(const cartesian_t &llh, cartesian_t &out) const {
	
    double *ccllh,*ccout;

    //Allocate arrays and transfer data through unified memory
    cudaMallocManaged(&ccllh,3*sizeof(double)); 
    cudaMallocManaged(&ccout,3*sizeof(double));

    
    cudaDeviceSynchronize();
    for (int i=0; i<3; i++) {
      ccllh[i]=llh[i];
    }

    //Call the global function with a single thread
    //forwardg<<<1,1>>>(ccllh,ccout,isnorth,akm1,e,lon0);
    
    //Get the results back
    cudaDeviceSynchronize();
    
    for (int i=0; i<3; i++) {
      out[i]=ccout[i];
    }
    
    //Clean up
    cudaFree(ccllh);
    cudaFree(ccout);
    
    return 0;
}



//
//__global__ void PolarStero::inverse_g(double *ups, double *llh)
//{
// 
//    /*
//     * Just a wrapper to call the device inverse function from host
//     */
//     inverse(ups,llh);
//  
//}

__host__ int PolarStereo::inverse(const cartesian_t &ups, cartesian_t &llh) const {

    /*
     * Host inverse projection function
     */	

    /*
     * Transform from Polar Stereo to LLH.
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

__device__ int PolarStereo::inverse(double ups[], double llh[]) const {

    /*
     * Device inverse projection function
     */	

    /*
     * Transform from Polar Stereo to LLH.
     * CUDA device function
     * TODO: organize shareable variables
     * TODO: find out how many iterations allow getting rid of the conditional,
     * replaced by a constant number of iterations.
     * TODO: test for numerical exceptions and out of bounds coordinates
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



int PolarStereo::inverse_h_single(const cartesian_t &out, cartesian_t &llh) const {
	

    double *ccups,*ccllh;
    int *ccret;

    //Allocate arrays and transfer data through unified memory
    cudaMallocManaged(&ccups,3*sizeof(double)); 
    cudaMallocManaged(&ccllh,3*sizeof(double));
    cudaMallocManaged(&ccret,sizeof(int));

    for (int i=0; i<3; i++) {
      ccups[i]=llh[i];
    }
    
    //Call the global function with a single thread 
    //inverseg<<<1,1>>>(ccups,ccllh);
    
    //Get the results back
    cudaDeviceSynchronize();
    
    for (int i=0; i<3; i++) {
      llh[i]=ccllh[i];
    }
    int ret=*ccret;
    
    //Clean up
    cudaFree(ccups);
    cudaFree(ccllh);
    cudaFree(ccret);
    
    return ret;
}

//__global__ void PolarStereo::roundtripcaller(int N, double *x, double *y, int *r, bool isnorth, double akm1, double e, double lon0){
//
//int tid=blockIdx.x*blockDim.x*blockDim.y*blockDim.z+threadIdx.z*blockDim.y*blockDim.x+threadIdx.y*blockDim.x+threadIdx.x;
//
//x[tid*3]=3.083894546782417e-02;
//x[tid*3+1]=-1.344622005845314e+00;
//x[tid*3+2]=1.912700155961942e+03;
//r[tid]=0;
//
//forward(x+tid*3,y+tid*3);
//inverse(y+tid*3,x+tid*3);
//
//}

int PolarStereo::roundtriptest(int np) const {

//TODO: Replace with proper sizing using deviceQuery() and memory
//footprint estimates
int maxt=1024;
int by=512;
int bx=np/(maxt*by);
//int N=1024*1024*512;
int N=np;
dim3 blocks( bx, by, 1 );
dim3 threads( maxt, 1, 1 );


cout<<"N="<<N<<endl;
cout<<"bx="<<bx<<" by="<<by<<" maxt="<<maxt<<endl;

cudaSetDevice(3);

//Allocate device arrays
double *x,*y;
int *r;
cudaMallocManaged(&x,3*N*sizeof(double));
cudaMallocManaged(&y,3*N*sizeof(double));
cudaMallocManaged(&r,N*sizeof(int));  

//cudaDeviceSynchronize();


cout.precision(17);
cout<<y[0]<< " " << y[1] << " " << y[2] << endl;
cout<<x[0]<< " " << x[1] << " " << x[2] << endl;

//Call the driver with many threads
//roundtripcaller<<<blocks,threads>>>(N,x,y,r,isnorth,akm1,e,lon0);

//Get results back
cudaDeviceSynchronize();

cout.precision(17);
cout<<y[0]<< " " << y[1] << " " << y[2] << endl;
cout<<x[0]<< " " << x[1] << " " << x[2] << endl;
cout<<endl;

//Clean up
cudaFree(x);
cudaFree(y);
cudaFree(r);
return 0;

}



/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* * * * * * * * * * * * * * * * * * * Projection Transformer * * * * * * * * * * * * * * * * * * */
int projTransform(ProjectionBase &in, ProjectionBase &out, const cartesian_t &inpts,
                  cartesian_t &outpts) {
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
        cartesian_t temp;
        if (in.inverse(inpts, temp) != 0) return -2;
        if (out.forward(temp, outpts) != 0) return 2;
    }
    return 0;
};
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
