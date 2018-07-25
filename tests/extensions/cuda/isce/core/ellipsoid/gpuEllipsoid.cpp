// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright 2018
//

#include <cmath>
#include <iostream>
#include <vector>
#include "isce/core/Constants.h"
#include "isce/core/Ellipsoid.h"
#include "isce/core/cuda/gpuEllipsoid.h"
#include "gtest/gtest.h"

using isce::core::Ellipsoid;
using isce::core::cuda::gpuEllipsoid;
using std::endl;
using std::vector;

//Some commonly used values
Ellipsoid wgs84_cpu(6378137.0, 0.0066943799901);
const double a_cpu = wgs84_cpu.a();
const double b_cpu = a_cpu * std::sqrt(1.0 - wgs84_cpu.e2());
gpuEllipsoid wgs84_gpu(6378137.0, 0.0066943799901);
const double a_gpu = wgs84_gpu.a;
const double b_gpu = a_gpu * std::sqrt(1.0 - wgs84_gpu.e2);

struct GpuEllipsoidTest : public ::testing::Test {
    virtual void SetUp() {
        fails = 0;
    }
    virtual void TearDown() {
        if (fails > 0) {
            std::cerr << "gpuEllipsoid::TearDown sees failures" << std::endl;
        }
    }
    unsigned fails;
};

#define ellipsoidGpuTest(name,p,q,r,x,y,z)       \
    TEST_F(GpuEllipsoidTest, name) {       \
        isce::core::cartesian_t ref_llh{p,q,r};    \
        isce::core::cartesian_t ref_xyz = {x,y,z};    \
        isce::core::cartesian_t xyz, gpu_xyz, cpu_xyz; \
        isce::core::cartesian_t llh, gpu_llh, cpu_llh; \
        llh = ref_llh;                  \
        wgs84_gpu.latLonToXyz_h(llh, gpu_xyz);    \
        wgs84_cpu.lonLatToXyz(llh, cpu_xyz);    \
        EXPECT_NEAR(gpu_xyz[0], ref_xyz[0], 1.0e-6);\
        EXPECT_NEAR(gpu_xyz[1], ref_xyz[1], 1.0e-6);\
        EXPECT_NEAR(gpu_xyz[2], ref_xyz[2], 1.0e-6);\
        EXPECT_NEAR(gpu_xyz[0], cpu_xyz[0], 1.0e-6);\
        EXPECT_NEAR(gpu_xyz[1], cpu_xyz[1], 1.0e-6);\
        EXPECT_NEAR(gpu_xyz[2], cpu_xyz[2], 1.0e-6);\
        xyz = ref_xyz;                  \
        wgs84_gpu.xyzToLatLon_h(xyz, gpu_llh);    \
        wgs84_cpu.xyzToLonLat(xyz, cpu_llh);    \
        EXPECT_NEAR(gpu_llh[0], ref_llh[0], 1.0e-9);\
        EXPECT_NEAR(gpu_llh[1], ref_llh[1], 1.0e-9);\
        EXPECT_NEAR(gpu_llh[2], ref_llh[2], 1.0e-6);\
        EXPECT_NEAR(gpu_llh[0], cpu_llh[0], 1.0e-9);\
        EXPECT_NEAR(gpu_llh[1], cpu_llh[1], 1.0e-9);\
        EXPECT_NEAR(gpu_llh[2], cpu_llh[2], 1.0e-6);\
        fails += ::testing::Test::HasFailure();\
    }    

ellipsoidGpuTest(Origin, {0.,0.,0.}, {a_cpu,0.,0.});

ellipsoidGpuTest(Equator90E, {0.5*M_PI, 0., 0.}, {0.,a_cpu,0.});

ellipsoidGpuTest(Equator90W,{-0.5*M_PI,0.,0.}, {0.,-a_cpu,0.});

ellipsoidGpuTest(EquatorDateline, {M_PI,0.,0.}, {-a_cpu,0.,0.});

ellipsoidGpuTest(NorthPole, {0.,0.5*M_PI,0.}, {0.,0.,b_cpu});

ellipsoidGpuTest(SouthPole, {0.,-0.5*M_PI,0.}, {0.,0.,-b_cpu});


ellipsoidGpuTest(Point1, {1.134431523585921e+00,-1.180097204507889e+00,7.552767636707697e+03},
        {1030784.925758840050548,2210337.910070449113846,-5881839.839890958741307});

ellipsoidGpuTest(Point2, {-1.988929481271171e+00,-3.218156967477281e-01,4.803829875484664e+02},
        {-2457926.302319798618555,-5531693.075449729338288,-2004656.608288598246872});

ellipsoidGpuTest(Point3, { 3.494775870065641e-01,1.321028021250511e+00, 6.684702668405185e+03},
        {1487474.649522442836314,542090.182021118933335, 6164710.02066358923912});

ellipsoidGpuTest(Point4, { 1.157071150199438e+00,1.539241336260909e+00,  2.075539115269004e+03},
        {81196.748833858233411,   184930.081202651723288, 6355641.007061666809022});

ellipsoidGpuTest(Point5, { 2.903217190227029e+00,3.078348660646868e-02, 1.303664510818545e+03},
          {-6196130.955770593136549,  1505632.319945097202435,195036.854449656093493});

ellipsoidGpuTest(Point6, { 1.404003364812063e+00,9.844570757478284e-01, 1.242074588639294e+03},
    {587386.746772550744936,  3488933.817566382698715, 5290575.784156281501055});

ellipsoidGpuTest(Point7, {1.786087533202875e+00,-1.404475795144668e+00,  3.047509859826395e+03},
        {-226426.343401445570635,  1035421.647801387240179, -6271459.446578867733479});

ellipsoidGpuTest(Point8, { -1.535570572315143e+00,-1.394372375292064e+00, 2.520818495701064e+01},
        {39553.214744714961853, -1122384.858932408038527, -6257455.705907705239952});

ellipsoidGpuTest(Point9, { 2.002720719284312e+00,-6.059309705813630e-01, -7.671870434220574e+01},
        {-2197035.039946643635631,  4766296.481927301734686, -3612087.398071805480868});

ellipsoidGpuTest(Point10, { -2.340221964131008e-01,1.162119493774084e+00,  6.948177664180818e+03},
         {2475217.167525716125965,  -590067.244431337225251, 5836531.74855871964246 });

ellipsoidGpuTest(Point11, {6.067080997777370e-01,-9.030342054807169e-01, 4.244471400804430e+02},
        {3251592.655810729600489,  2256703.30570419318974 ,-4985277.930962197482586});

ellipsoidGpuTest(Point12, { -2.118133740176279e+00,9.812354487540356e-01, 2.921301812478523e+03},
         {-1850635.103680874686688, -3036577.247930331621319,5280569.380736761726439});

ellipsoidGpuTest(Point13, { -2.005023821660764e+00,1.535487121535718e+00, 2.182275729585851e+02},
         { -95048.576977927994449,  -204957.529435861855745, 6352981.530775795690715});

ellipsoidGpuTest(Point14, {2.719747828172381e+00,-1.552548149921413e+00,  4.298201230045657e+03},
        {-106608.855637043248862,    47844.679874961388123, -6359984.3118050172925});

ellipsoidGpuTest(Point15, { -1.498660315787147e+00,1.076512019764726e+00, 8.472554905622580e+02},
         {218676.696484291809611, -3026189.824885316658765, 5592409.664520519785583});

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
