/**
 * 
 * Benchmark tests for gpu polar projection forward/inverse functions
 * These are intended only to test for speedup of transformations done on the device.
 * See gpuPolar.cpp for unit tests, where single point transformations are compared to tabulated
 * results, to test correctness of device conversions.
 * Transformations use many millions of points are transformed
 * in parallel in gpu, and the wall times are compared between cpu and gpu code.
 * The number of tests can be controlled by an environment variable, gpubenchmark_ntests.
 * If gpubenchmark_ntests is not set, ntests defaults to 4.
 * Tests start at 1M points, each successive test doubles the number of points to use.
 * Results for gpu and cpu times are printed to stdout, as CSV.
 * Note that the first test always takes a very long time on gpu, due to the time to acquire and
 * initialize the device. So I mostly disregard the times for the first test case.
 * 
 *
 *
 * Source Author: Paulo Penteado
 * Copyright 2018
 */

#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream> 
#include "isce/cuda/core/gpuProjections.h"
#include "isce/core/Projections.h"
#include <chrono>

using isce::core::cartesian_t;
using std::cout;
using std::endl;
using std::getenv;



double roundtriptest(int np, isce::core::PolarStereo hemi) {
int N=np;
//cout<<"N="<<N<<endl;
double *x,*y;
int *r;

x=new double [3*N];
y=new double [3*N];
r=new int [3*N];

cartesian_t xyz,llh;
//cout.precision(17);
//cout<<llh[0]<< " " << llh[1] << " " << llh[2] << endl;
//cout<<xyz[0]<< " " << xyz[1] << " " << xyz[2] << endl;

double x0[3];
/**
 * { 3.083894546782417e-02,-1.344622005845314e+00, 1.912700155961942e+03} maps to { 4.359317146290433e+04, 1.413127144940603e+06, 1.912700155961942e+03}
 * {-2.748054994702517e+00,-1.475217756400627e+00,-1.287407634430560e+02} maps to {-2.283273518457955e+05,-5.499261910668013e+05,-1.287407634430560e+02}
 */
for (int tid=0; tid<N; tid++) {
x[tid*3]=3.083894546782417e-02+(-2.748054994702517e+00-3.083894546782417e-02)*((double) tid)/((double) (N-1));
x[tid*3+1]=-1.344622005845314e+00+(-1.475217756400627e+00+1.344622005845314e+00)*((double) tid)/((double) (N-1));
x[tid*3+2]=1.912700155961942e+03+(-1.287407634430560e+02-1.912700155961942e+03)*((double) tid)/((double) (N-1));
r[tid]=0;
llh={x[tid*3],x[tid*3+1],x[tid*3+2]};
for (int i=0; i<3; i++){
	   x0[i]=x[tid*3+i];
}
hemi.forward(llh, xyz);    
hemi.inverse(xyz, llh);
r[tid]=sqrt(pow(x0[0]-x[tid*3],2.)+pow(x0[1]-x[tid*3+1],2.));
}

//cout.precision(17);
//cout<<xyz[0]<< " " << xyz[1] << " " << xyz[2] << endl;
//cout<<llh[0]<< " " << llh[1] << " " << llh[2] << endl;
//cout<<"---"<<endl;

double maxd=r[0];
/** For testing roundtrip correctness, calculate maximum difference between input and output latlon
 *  For benchmarking, since this operation should not be included in timing, it is commented out, and
 *  just the first difference (r[0]) is reported instead.
 */
//maxd=*std::max_element(r,r+N-1);


//Clean up
delete[] x;
delete[] y;
delete[] r;

return maxd;
}


int main(int argc, char **argv) {

	using clock = std::chrono::steady_clock;
	clock::time_point start,end;
	std::chrono::duration<double> elapsedg,elapsedc;
	
	isce::cuda::core::PolarStereo gSouth(3031);
	isce::core::PolarStereo South(3031);
	isce::cuda::core::PolarStereo gNorth(3413);
	isce::core::PolarStereo North(3413);

	
	int np;
	struct count{
		int np;
		double gputime;
		double cputime;
	};
	count counts[10];
	np=1024*1024;
	int ntests=0;
	char* env_nt=getenv("gpubenchmark_ntests");
	if (env_nt) {
		std::stringstream envs(env_nt);
		envs>>ntests;
	} else {
		ntests=4;
	}
	
	cout<<"Running gpu/cpu benchmarks for "<<ntests<<" cases"<<endl;
	cout<<"(ntests can be set with environment variable gpubenchmark_ntests)"<<endl;
	cout<<"nt,np,gputime,cputime,maxdg,maxdc"<<endl;
	double maxdg,maxdc;
	for (int sz=0; sz<ntests; sz++){
		start=clock::now();	
		maxdg=gSouth.roundtriptest(np);
		end=clock::now();
		elapsedg=end-start;
		//cout<<elapsedg.count()<<endl;
		
		//cout<<"rt tests done\n";
		start=clock::now();
		maxdc=roundtriptest(np,South);
		end=clock::now();
		elapsedc=end-start;
		//cout<<"rt tests done\n"; 
		//cout<<elapsedc.count()<<endl;
		counts[sz]={np,elapsedg.count(),elapsedc.count()};
		np*=2;
		cout<<sz<<","<<counts[sz].np<<","<<counts[sz].gputime<<","<<counts[sz].cputime<<","<<maxdg<<","<<maxdc<<endl;
	}

    
    return 0;
}
