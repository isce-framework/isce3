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
CUDA_HOSTDEV double pj_tsfn(double phi, double sinphi, double e) {
    /*
     * Local function - Determine small t from PROJ.4.
     */
    sinphi *= e;
    return tan(.5 * ((.5*M_PI) - phi)) / pow((1. - sinphi) / (1. + sinphi), .5*e);
}
}}}




    CUDA_HOSTDEV PolarStereo::PolarStereo(int code)  {
    /**
     * Constructor for the class, based on EPSG projection code
     * Set up various parameters for polar stereographic projection. Currently only EPSG:3031
     * (Antarctic) and EPSG:3413 (Greenland) are supported.
     */
	   
	ellipse=gpuEllipsoid(6378137.,.0066943799901);
	_epsgcode=code;
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
    } else { //TODO: error handling when running as a __device__ function
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




  CUDA_GLOBAL void forward_g(PolarStereo ps, double* llh, double * out) {
    /**
     * Just a wrapper to call the device forward function from host for a single conversion
     * Needs to be a global function, due to CUDA requirements
     */
	 
     ps.forward(llh,out);
  
  }
   

  CUDA_HOSTDEV int PolarStereo::forward(double llh[], double out[])  const{
        

    /**
     * Host / Device forward projection function.
     * Must use double* instead of cartesian_t, because cartesian_t (std::array)
     * is not available for __device__ functions.
     */	
	
	
	
    /**
     * Transform from LLH to Polar Stereo.
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

  CUDA_HOST int PolarStereo::forward(const cartesian_t &llh, cartesian_t &out) const {
	  
   /**
	* Wrapper to call device function on a single thread to do a single coordinate conversion
	* on the device, from host. To do a single coordinate conversion on host, one can use the
	* forward method with doubles instead of cartesian_t: forward(double llh[], double out[])
	* is a __host__ __device__ function.
	*/
	
    double *ccllh,*ccout;

    //Allocate arrays and transfer data through unified memory
    cudaMallocManaged(&ccllh,3*sizeof(double)); 
    cudaMallocManaged(&ccout,3*sizeof(double));

    
    cudaDeviceSynchronize();
    for (int i=0; i<3; i++) {
      ccllh[i]=llh[i];
    }

    //Call the global function with a single thread
    forward_g<<<1,1>>>(*this,ccllh,ccout);
    
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




  CUDA_GLOBAL void inverse_g(PolarStereo ps, double *ups, double *llh){
 
	/**
	 * Just a wrapper to call the device inverse function from host for a single conversion
	 * Needs to be a global function, due to CUDA requirements
	 */
     ps.inverse(ups,llh);
  
  }


  CUDA_HOSTDEV int PolarStereo::inverse(double ups[], double llh[]) const {

	/**
	 * Host / Device inverse projection function.
	 * Must use double* instead of cartesian_t, because cartesian_t (std::array)
	 * is not available for __device__ functions.
	 */	

    /**
     * Transform from Polar Stereo to LLH.
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



  CUDA_HOST int PolarStereo::inverse(const cartesian_t &ups, cartesian_t &llh) const {
	  
    /**
	 * Wrapper to call device function on a single thread to do a single coordinate conversion
	 * on the device, from host. To do a single coordinate conversion on host, one can use the
	 * inverse method with doubles instead of cartesian_t: inverse(double llh[], double out[])
	 * is a __host__ __device__ function.
	 */

    double *ccups,*ccllh;
    int *ccret;

    //Allocate arrays and transfer data through unified memory
    cudaMallocManaged(&ccups,3*sizeof(double)); 
    cudaMallocManaged(&ccllh,3*sizeof(double));
    cudaMallocManaged(&ccret,sizeof(int));

    for (int i=0; i<3; i++) {
      ccups[i]=ups[i];
    }
    
    //Call the global function with a single thread 
    inverse_g<<<1,1>>>(*this,ccups,ccllh);

    
    
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

  CUDA_GLOBAL void roundtripcaller(PolarStereo ps, int N, double *x, double *y, double *r){

	  
   /**
    * Wrapper to call the device forward and inverse functions from host for many round trip conversions,
    * for testing and benchmarks.
    * Needs to be a global function, due to CUDA requirements
    */

   int tid=blockIdx.x*blockDim.x*blockDim.y*blockDim.z+threadIdx.z*blockDim.y*blockDim.x+threadIdx.y*blockDim.x+threadIdx.x;
   /**
    * { 3.083894546782417e-02,-1.344622005845314e+00, 1.912700155961942e+03} maps to { 4.359317146290433e+04, 1.413127144940603e+06, 1.912700155961942e+03}
    * {-2.748054994702517e+00,-1.475217756400627e+00,-1.287407634430560e+02} maps to {-2.283273518457955e+05,-5.499261910668013e+05,-1.287407634430560e+02}
    */
   x[tid*3]=3.083894546782417e-02+(-2.748054994702517e+00-3.083894546782417e-02)*((double) tid)/((double) (N-1));
   x[tid*3+1]=-1.344622005845314e+00+(-1.475217756400627e+00+1.344622005845314e+00)*((double) tid)/((double) (N-1));
   x[tid*3+2]=1.912700155961942e+03+(-1.287407634430560e+02-1.912700155961942e+03)*((double) tid)/((double) (N-1));
   r[tid]=0.;
   double x0[3];
   for (int i=0; i<3; i++){
	   x0[i]=x[tid*3+i];
   }

   ps.forward(x+tid*3,y+tid*3);
   ps.inverse(y+tid*3,x+tid*3);
   r[tid]=sqrt(pow(x0[0]-x[tid*3],2.)+pow(x0[1]-x[tid*3+1],2.));

   

  }
  


  CUDA_HOST double PolarStereo::roundtriptest(int np) {
	  
	  
	/**
	 * Benchmark method to perform np calls to the device forward and inverse functions
	 * from host for many round trip conversions.
	 * 
	 */
	  

    //TODO: Replace with proper sizing using deviceQuery() and memory
    //footprint estimates
	  
   int cudadev=0;
   cudaSetDevice(cudadev);
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, cudadev);

	  
   int maxt=1024;
   int bz=512;
   /**
    * General case: figure out thread/block sizes based on device properties
    * Should be moved to something done at compile time, because these API calls
    * take some time, thus hurt the benchmarks a bit
    */
   maxt=prop.maxThreadsPerBlock;
   bz=std::min({512,prop.maxThreadsDim[0]});
   int by=np/(maxt*bz);
   int N=np;
   dim3 blocks( 1, by, bz );
   dim3 threads( maxt, 1, 1 );


   //cout<<"N="<<N<<endl;
   //cout<<"bz="<<bz<<" by="<<by<<" maxt="<<maxt<<endl;


   //Allocate device arrays
   double *x,*y, *r;
   
   cudaMallocManaged(&x,3*N*sizeof(double));
   cudaMallocManaged(&y,3*N*sizeof(double));
   cudaMallocManaged(&r,N*sizeof(double));  

   //cudaDeviceSynchronize();


   //cout.precision(17);
   //cout<<y[0]<< " " << y[1] << " " << y[2] << endl;
   //cout<<x[0]<< " " << x[1] << " " << x[2] << endl;

   //Call the driver with many threads
   roundtripcaller<<<blocks,threads>>>(*this,N,x,y,r);

   //Get results back
   /** For testing roundtrip time without memory transfer time, comment next line
    */
   //cudaDeviceSynchronize();
   
   double maxd=r[0];
   /** For testing roundtrip correctness, calculate maximum difference between input and output latlon
    *  For benchmarking, since this operation should not be included in timing, it is commented out, and
    *  just the first difference (r[0]) is reported instead.
    */
   //maxd=*std::max_element(r,r+N-1);
   
   //cout<<"maxd:"<<*maxd<<endl;

   //cout<<y[0]<< " " << y[1] << " " << y[2] << endl;
   //cout<<x[0]<< " " << x[1] << " " << x[2] << endl;
   //cout<<"===="<<endl;

   //Clean up
   cudaFree(x);
   cudaFree(y);
   cudaFree(r);
   return maxd;

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
