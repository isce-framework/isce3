//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Marco Lavalle
// Copyright 2018
//

#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <numeric>
#include <gtest/gtest.h>

#include "isce/core/Raster.h"

// Support function to check if file exists
inline bool exists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}


// Global variables
struct ProjTest : public ::testing::Test {
  const uint nc = 10;    // number of columns
  const uint nl = 15;    // number of lines
  const double x0 = -24000.; //x origin
  const double y0 = 10000.;  //y origin
  const double dx = 30.0;    //xspacing
  const double dy = -50.0;   //yspacing
  const std::string rawFilename = "test";
  const std::string projFilename = "test.vrt";
};


#define projTest(code, name) \
    TEST_F(ProjTest, name) { \
        std::remove(rawFilename.c_str()); \
        std::remove(projFilename.c_str()); \
        double trans[] = {x0, dx, 0., y0, 0., dy}; \
        std::valarray<double> transval(trans, 6); \
        std::vector<double> transvec(trans, trans+6); \
        int incode = code; \
        isce::core::Raster img = isce::core::Raster( projFilename, nc, nl, 1, GDT_Float32, "VRT" ); \
        img.setEPSG(incode); \
        img.setGeoTransform(trans); \
        img.setGeoTransform(transval); \
        img.setGeoTransform(transvec); \
        std::valarray<double> otransval(6); \
        std::vector<double> otransvec(6); \
        double otransarr[6]; \
        img.getGeoTransform(otransval); \
        img.getGeoTransform(otransvec); \
        img.getGeoTransform(otransarr); \
        ASSERT_EQ( img.x0(), x0 ); \
        ASSERT_EQ( img.y0(), y0 ); \
        ASSERT_EQ( img.dx(), dx ); \
        ASSERT_EQ( img.dy(), dy ); \
        ASSERT_EQ( img.getEPSG(), incode); \
    }

//Macro for name of the test
#define epsgTestName(ind) ind ## _EPSG
#define epsgTest(x) \
    projTest(x, epsgTestName(x));

//Test each of the UTM north zones
epsgTest(32601);
epsgTest(32602);
epsgTest(32603);
epsgTest(32604);
epsgTest(32605);
epsgTest(32606);
epsgTest(32607);
epsgTest(32608);
epsgTest(32609);
epsgTest(32610);
epsgTest(32611);
epsgTest(32612);
epsgTest(32613);
epsgTest(32614);
epsgTest(32615);
epsgTest(32616);
epsgTest(32617);
epsgTest(32618);
epsgTest(32619);
epsgTest(32620);
epsgTest(32621);
epsgTest(32622);
epsgTest(32623);
epsgTest(32624);
epsgTest(32625);
epsgTest(32626);
epsgTest(32627);
epsgTest(32628);
epsgTest(32629);
epsgTest(32630);
epsgTest(32631);
epsgTest(32632);
epsgTest(32633);
epsgTest(32634);
epsgTest(32635);
epsgTest(32636);
epsgTest(32637);
epsgTest(32638);
epsgTest(32639);
epsgTest(32640);
epsgTest(32641);
epsgTest(32642);
epsgTest(32643);
epsgTest(32644);
epsgTest(32645);
epsgTest(32646);
epsgTest(32647);
epsgTest(32648);
epsgTest(32649);
epsgTest(32650);
epsgTest(32651);
epsgTest(32652);
epsgTest(32653);
epsgTest(32654);
epsgTest(32655);
epsgTest(32656);
epsgTest(32657);
epsgTest(32658);
epsgTest(32659);
epsgTest(32660);


//Test each of the UTM south zones
epsgTest(32701);
epsgTest(32702);
epsgTest(32703);
epsgTest(32704);
epsgTest(32705);
epsgTest(32706);
epsgTest(32707);
epsgTest(32708);
epsgTest(32709);
epsgTest(32710);
epsgTest(32711);
epsgTest(32712);
epsgTest(32713);
epsgTest(32714);
epsgTest(32715);
epsgTest(32716);
epsgTest(32717);
epsgTest(32718);
epsgTest(32719);
epsgTest(32720);
epsgTest(32721);
epsgTest(32722);
epsgTest(32723);
epsgTest(32724);
epsgTest(32725);
epsgTest(32726);
epsgTest(32727);
epsgTest(32728);
epsgTest(32729);
epsgTest(32730);
epsgTest(32731);
epsgTest(32732);
epsgTest(32733);
epsgTest(32734);
epsgTest(32735);
epsgTest(32736);
epsgTest(32737);
epsgTest(32738);
epsgTest(32739);
epsgTest(32740);
epsgTest(32741);
epsgTest(32742);
epsgTest(32743);
epsgTest(32744);
epsgTest(32745);
epsgTest(32746);
epsgTest(32747);
epsgTest(32748);
epsgTest(32749);
epsgTest(32750);
epsgTest(32751);
epsgTest(32752);
epsgTest(32753);
epsgTest(32754);
epsgTest(32755);
epsgTest(32756);
epsgTest(32757);
epsgTest(32758);
epsgTest(32759);
epsgTest(32760);

//Test for polar stereographic zones
epsgTest(3031);
epsgTest(3413);

//Test for EASE grid
epsgTest(6933);

//Test for lat/lon
epsgTest(4326);

// Main
int main( int argc, char * argv[] ) {
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}


// end of file
