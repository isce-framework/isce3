//
// Source Author: Paulo Penteado, based on polar.cpp by Joshua Cohen
// Copyright 2018
//

#include <cmath>
#include <iostream>
#include "isce/extensions/gpuProjections.h"
#include "isce/core/Projections.h"
#include <chrono>

#include "gtest/gtest.h"
using isce::core::cartesian_t;
using std::cout;
using std::endl;



gpuisce::core::PolarStereo gNorth(3413);
gpuisce::core::PolarStereo gSouth(3031);


struct PolarTest : public ::testing::Test {
    virtual void SetUp() {
        fails = 0;
    }
    virtual void TearDown() {
        if (fails > 0) {
            std::cerr << "Polar::TearDown sees failures" << std::endl;
        }
    }
    unsigned fails;
};


#define polarTest(hemi,name,p,q,r,x,y,z)       \
    TEST_F(PolarTest, name) {       \
        cartesian_t ref_llh = {p,q,r};    \
        cartesian_t ref_xyz = {x,y,z};    \
        cartesian_t xyz, llh;  \
        llh = ref_llh;                  \
        hemi.forward(llh, xyz);    \
        EXPECT_NEAR(xyz[0], ref_xyz[0], 1.0e-6);\
        EXPECT_NEAR(xyz[1], ref_xyz[1], 1.0e-6);\
        EXPECT_NEAR(xyz[2], ref_xyz[2], 1.0e-6);\
        xyz = ref_xyz;                  \
        hemi.inverse(xyz, llh);    \
        EXPECT_NEAR(llh[0], ref_llh[0], 1.0e-9);\
        EXPECT_NEAR(llh[1], ref_llh[1], 1.0e-9);\
        EXPECT_NEAR(llh[2], ref_llh[2], 1.0e-6);\
        fails += ::testing::Test::HasFailure();\
    }



polarTest(gNorth, NorthPole3413, {0,0.5*M_PI, 0.}, {0.,0.,0.});

polarTest(gSouth, SouthPole3031, {0,-0.5*M_PI,0.}, {0.,0.,0.});


//South pole tests
polarTest(gSouth, South1, {3.083894546782417e-02,  -1.344622005845314e+00, 1.912700155961942e+03},
        {  4.359317146290433e+04,   1.413127144940603e+06, 1.912700155961942e+03});

polarTest(gSouth, South2, { -2.748054994702517e+00,  -1.475217756400627e+00, -1.287407634430560e+02},
        { -2.283273518457955e+05,  -5.499261910668013e+05,-1.287407634430560e+02});

polarTest(gSouth, South3, { -2.015678088976863e+00,  -1.055155908044944e+00, 1.092248256095492e+03},
        {-2.960932935986692e+06,  -1.411648585097893e+06,1.092248256095492e+03});

polarTest(gSouth, South4, {  9.897670707087629e-01,  -1.171084250110058e+00, 1.168298880338552e+03},
        { 2.107021273612706e+06,   1.383555506347802e+06, 1.168298880338552e+03});

polarTest(gSouth, South5, { -9.395682598011037e-01,  -1.312042832846017e+00, 1.956528789302522e+02},
        {-1.307435699257175e+06,   9.557320098285378e+05, 1.956528789302522e+02});

polarTest(gSouth, South6, {  3.059787626007144e+00,  -1.264032243692100e+00, 2.180598888525850e+03},
        { 1.572353168179760e+05,  -1.917784542305386e+06, 2.180598888525850e+03});

polarTest(gSouth, South7, { -9.916791410537584e-01,  -1.514697202241026e+00, -2.318063194460424e+02},
        {-2.923647622874998e+05,   1.911794872084717e+05, -2.318063194460424e+02});

polarTest(gSouth, South8, {  1.517418470018747e+00,  -1.391152237972874e+00, 2.446804499753949e+03},
        { 1.119641885143999e+06,   5.982090880356800e+04, 2.446804499753949e+03});

polarTest(gSouth, South9, { -2.182268380815120e+00,  -1.175078082197829e+00, 2.140275478786234e+03},
        {-2.042780119449563e+06,  -1.432218175292800e+06, 2.140275478786234e+03});

polarTest(gSouth, South10, {  2.634244174408007e+00,  -1.532775723363972e+00, 2.342510644576513e+03},
        { 1.150121609211143e+05,  -2.069000438839252e+05, 2.342510644576513e+03});

polarTest(gSouth, South11, { -2.803091043255459e+00,  -1.382761838119448e+00, 1.675322599947393e+03},
        {-3.898210801208878e+05,  -1.107283029746334e+06, 1.675322599947393e+03});

polarTest(gSouth, South12, { -2.528534168939039e+00,  -1.499361341379746e+00, -3.477556810227646e+02},
        {-2.559757493525218e+05,  -3.638700923671821e+05, -3.477556810227646e+02});

polarTest(gSouth, South13, {  5.857567927116056e-01,  -1.502224765955195e+00, 3.754108917322826e+02},
        { 2.360811101247164e+05,   3.558506213813419e+05, 3.754108917322826e+02});

polarTest(gSouth, South14, { -2.439357770938500e+00,  -1.442480697904947e+00, 1.469661241145656e+03},
        {-5.166500161522263e+05,  -6.106136854812840e+05, 1.469661241145656e+03});

polarTest(gSouth, South15, {  3.425915795274763e-01,  -1.340153122367854e+00, 1.586442146039017e+03},
        { 4.844001591422016e+05,   1.358174101228337e+06, 1.586442146039017e+03});


//North polar tests
polarTest(gNorth, North1, { 1.526573781702310e+00,   1.204871941981861e+00, 8.667190129005166e+02},
        { 1.693588103815650e+06,   1.550052481860258e+06, 8.667190129005166e+02});

polarTest(gNorth, North2, { 1.710130591777928e+00,   1.259029134979930e+00, -2.475029443026675e+02},
        {  1.174137939572316e+06,   1.557187175159873e+06, -2.475029443026675e+02});

polarTest(gNorth, North3, {-2.160034946819783e+00,   1.551411899481004e+00, 1.735710069735323e+03},
         {-1.180095090698630e+05,  -2.345024306168611e+04, 1.735710069735323e+03});

polarTest(gNorth, North4, {-2.901136118931835e+00,   1.379807749008712e+00, 2.298788309982012e+01},
         {-1.016679109893707e+06,   6.162771930043071e+05, 2.298788309982012e+01});

polarTest(gNorth, North5, {-2.436023620801858e+00,   1.425257192961883e+00, 1.376274653617448e+03},
        {-9.019692513983058e+05,   7.215676389938517e+04, 1.376274653617448e+03});

polarTest(gNorth, North6, { 1.068629494000629e+00,   1.416306900974420e+00, 2.040382460175913e+03},
        { 9.224269061442432e+05,   2.684779897850138e+05, 2.040382460175913e+03});

polarTest(gNorth, North7, {-1.568146909395947e+00,   1.358441316018295e+00, 1.915165642793210e+02},
        {-9.328784669868136e+05,  -9.378347788054653e+05, 1.915165642793210e+02});

polarTest(gNorth, North8, { 2.001037214400984e+00,   1.477207655719103e+00, 2.426120691364692e+03},
        { 2.021343869147930e+05,   5.450069830147136e+05, 2.426120691364692e+03});

polarTest(gNorth, North9, {-2.930378328448844e+00,   1.103512555140625e+00, 2.470915925404410e+03},
        {-2.478697264190398e+06,   1.603437501974491e+06, 2.470915925404410e+03});

polarTest(gNorth, North10, {-1.549395073358587e+00,   1.118979154520375e+00, 8.703932224186308e+02},
        {-1.972404065438736e+06,  -2.058687654382347e+06, 8.703932224186308e+02});

polarTest(gNorth, North11, {-2.109013419166581e+00,   1.564381687922278e+00, 1.610579390463871e+02},
        {-3.860375611966997e+04,  -9.741323985686075e+03, 1.610579390463871e+02});

polarTest(gNorth, North12, {-2.418990503089881e+00,   1.193662955967620e+00, 1.186164193787148e+03},
         {-2.363108981136872e+06,   1.485891859449324e+05, 1.186164193787148e+03});

polarTest(gNorth, North13, { 1.101658547536258e+00,   1.082647115833457e+00, 1.126494838546805e+03},
        { 2.935798960966537e+06,   9.607232598177509e+05, 1.126494838546805e+03});

polarTest(gNorth, North14, {-2.044058884889994e+00,   1.467509794204064e+00, 1.355282188196872e+03},
        {-6.106128932646384e+05,  -1.970349252242279e+05, 1.355282188196872e+03});

polarTest(gNorth, North15, { 1.896312874775227e+00,   1.411713711367596e+00, 1.689205800030411e+03},
        { 4.391289593706741e+05,   8.865894956770649e+05, 1.689205800030411e+03});



int roundtriptest(int np, isce::core::PolarStereo hemi) {
int N=np;
cout<<"N="<<N<<endl;
//Allocate device arrays
double *x,*y;
int *r;

x=new double [3*N];
y=new double [3*N];
r=new int [3*N];

cout.precision(17);
cout<<y[0]<< " " << y[1] << " " << y[2] << endl;
cout<<x[0]<< " " << x[1] << " " << x[2] << endl;

//Call the driver with many threads
cartesian_t xyz,llh;
for (int tid=0; tid<N; tid++) {
x[tid*3]=3.083894546782417e-02;
x[tid*3+1]=-1.344622005845314e+00;
x[tid*3+2]=1.912700155961942e+03;
r[tid]=0;
llh={x[tid*3],x[tid*3+1],x[tid*3+2]};
hemi.forward(llh, xyz);    
hemi.inverse(xyz, llh);
}

cout.precision(17);
cout<<y[0]<< " " << y[1] << " " << y[2] << endl;
cout<<x[0]<< " " << x[1] << " " << x[2] << endl;
cout<<endl;

//Clean up
delete[] x;
delete[] y;
delete[] r;

return 0;
}


int main(int argc, char **argv) {

	using clock = std::chrono::steady_clock;
	clock::time_point start,end;
	std::chrono::duration<double> elapsedg,elapsedc;
	
	gpuisce::core::PolarStereo gSouth(3031);
	isce::core::PolarStereo South(3031);
	int np;
	np=1024*1024*128;
	struct count{
		int np;
		double gputime;
		double cputime;
	};
	count counts[10];
	np=1024*1024;
	for (int sz=0; sz<10; sz++){
		start=clock::now();	
		//for (int i=0; i<2; i++) {
		  gSouth.roundtriptest(np);
		//}
		end=clock::now();
		elapsedg=end-start;
		//cout<<elapsedg.count()<<endl;
		
		//cout<<"rt tests done\n";
		start=clock::now();
		//for (int i=0; i<2; i++) {
		  roundtriptest(np,South);
		//}
		end=clock::now();
		elapsedc=end-start;
		//cout<<"rt tests done\n"; 
		//cout<<elapsedc.count()<<endl;
		counts[sz]={np,elapsedg.count(),elapsedc.count()};
		np*=2;
		cout<<counts[sz].np<<","<<counts[sz].gputime<<","<<counts[sz].cputime<<endl;
	}

    //::testing::InitGoogleTest(&argc, argv);
    
    //return RUN_ALL_TESTS();
}
