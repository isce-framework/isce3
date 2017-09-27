//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <iostream>
#include <vector>
#include "isce/core/Constants.h"
#include "isce/core/Ellipsoid.h"
using isce::core::Ellipsoid;
using isce::core::LLH_2_XYZ;
using isce::core::XYZ_2_LLH;
using std::cout;
using std::endl;
using std::vector;

bool checkAlmostEqual(vector<double> &ref, vector<double> &calc, int n_digits) {
    /*
     *  Calculate if two vectors are almost equal to n_digits of precision.
     */

    bool stat = true;
    for (int i=0; i<ref.size(); i++) {
        stat = stat & (abs(ref[i] - calc[i]) < pow(10., -n_digits));
    }
    if (!stat) {
        cout << "    Error:" << endl;
        cout << "    Expected [" << ref[0] << ", " << ref[1] << ", " << ref[2] << "]" << endl;
        cout << "    Received [" << calc[0] << ", " << calc[1] << ", " << calc[2] << "]" << endl;
    }
    return stat;
}

void testCorners() {
    /*
     * Test corners for lat/lon
     */

    Ellipsoid wgs84(6378137.0, 0.0066943799901);
    double b = wgs84.a * std::sqrt(1 - wgs84.e2);
    double a = wgs84.a;

    cout << endl << " [Origin]" << endl;
    {
        vector<double> ref_llh = {0.,0.,0.};
        vector<double> ref_xyz = {a,0.,0.};
        vector<double> xyz(3), llh(3);

        llh = ref_llh;
        wgs84.latLonToXyz(xyz, llh);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 9);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.xyzToLatLon(xyz, llh);
        stat = checkAlmostEqual(llh, ref_llh,9);
        cout << " [XYZ_2_LLH] ";
        if(stat) cout << "PASSED";
        cout << endl;
    }

    cout << endl << " [90-degrees]" << endl;
    {
        vector<double> ref_llh = {0.,0.5*M_PI,0.};
        vector<double> ref_xyz = {0.,a,0.};
        vector<double> xyz(3),llh(3);

        llh = ref_llh;
        wgs84.latLonToXyz(xyz, llh);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 9);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.xyzToLatLon(xyz, llh);
        stat = checkAlmostEqual(llh, ref_llh,9);
        cout << " [XYZ_2_LLH] ";
        if(stat) cout << "PASSED";
        cout << endl;
    }

    cout << endl << " [270-degrees]" << endl;
    {
        vector<double> ref_llh = {0.,-0.5*M_PI,0.};
        vector<double> ref_xyz = {0.,-a,0.};
        vector<double> xyz(3),llh(3);

        llh = ref_llh;
        wgs84.latLonToXyz(xyz, llh);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 9);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.xyzToLatLon(xyz, llh);
        stat = checkAlmostEqual(llh, ref_llh,9);
        cout << " [XYZ_2_LLH] ";
        if(stat) cout << "PASSED";
        cout << endl;
    }

    cout << endl << " [180-degrees]" << endl;
    {
        vector<double> ref_llh = {0.,M_PI,0.};
        vector<double> ref_xyz = {-a,0.,0.};
        vector<double> xyz(3),llh(3);

        llh = ref_llh;
        wgs84.latLonToXyz(xyz, llh);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 9);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.xyzToLatLon(xyz, llh);
        stat = checkAlmostEqual(llh, ref_llh,9);
        cout << " [XYZ_2_LLH] ";
        if(stat) cout << "PASSED";
        cout << endl;
    }

    cout << endl << " [North pole]" << endl;
    {
        vector<double> ref_llh = {0.5*M_PI,0.,0.};
        vector<double> ref_xyz = {0.,0.,b};
        vector<double> xyz(3),llh(3);

        llh = ref_llh;
        wgs84.latLonToXyz(xyz, llh);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 9);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.xyzToLatLon(xyz, llh);
        stat = checkAlmostEqual(llh, ref_llh,9);
        cout << " [XYZ_2_LLH] ";
        if(stat) cout << "PASSED";
        cout << endl;
    }

    cout << endl << " [South pole]" << endl;
    {
        vector<double> ref_llh = {-0.5*M_PI,0.,0.};
        vector<double> ref_xyz = {0.,0.,-b};
        vector<double> xyz(3),llh(3);

        llh = ref_llh;
        wgs84.latLonToXyz(xyz, llh);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 9);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.xyzToLatLon(xyz, llh);
        stat = checkAlmostEqual(llh, ref_llh,9);
        cout << " [XYZ_2_LLH] ";
        if(stat) cout << "PASSED";
        cout << endl;
    }

}

void testCoords() {
    /*
     * Test corners for lat/lon
     */

    Ellipsoid wgs84(6378137.0, 0.0066943799901);
    
    //Test data was generated using pyproj and random numbers
    double ref_llh[15][3] = {{ -1.180097204507889e+00,   1.134431523585921e+00,
          7.552767636707697e+03},
       {-3.218156967477281e-01,  -1.988929481271171e+00,
          4.803829875484664e+02},
       { 1.321028021250511e+00,   3.494775870065641e-01,
          6.684702668405185e+03},
       { 1.539241336260909e+00,   1.157071150199438e+00,
          2.075539115269004e+03},
       { 3.078348660646868e-02,   2.903217190227029e+00,
          1.303664510818545e+03},
       { 9.844570757478284e-01,   1.404003364812063e+00,
          1.242074588639294e+03},
       {-1.404475795144668e+00,   1.786087533202875e+00,
          3.047509859826395e+03},
       {-1.394372375292064e+00,  -1.535570572315143e+00,
          2.520818495701064e+01},
       {-6.059309705813630e-01,   2.002720719284312e+00,
         -7.671870434220574e+01},
       { 1.162119493774084e+00,  -2.340221964131008e-01,
          6.948177664180818e+03},
       {-9.030342054807169e-01,   6.067080997777370e-01,
          4.244471400804430e+02},
       { 9.812354487540356e-01,  -2.118133740176279e+00,
          2.921301812478523e+03},
       { 1.535487121535718e+00,  -2.005023821660764e+00,
          2.182275729585851e+02},
       {-1.552548149921413e+00,   2.719747828172381e+00,
          4.298201230045657e+03},
       { 1.076512019764726e+00,  -1.498660315787147e+00,
          8.472554905622580e+02}};
    
    double ref_xyz[15][3] = {{ 1030784.925758840050548,  2210337.910070449113846,
        -5881839.839890958741307},
       {-2457926.302319798618555, -5531693.075449729338288,
        -2004656.608288598246872},
       {1487474.649522442836314,   542090.182021118933335,
         6164710.02066358923912 },
       {  81196.748833858233411,   184930.081202651723288,
         6355641.007061666809022},
       {-6196130.955770593136549,  1505632.319945097202435,
          195036.854449656093493},
       { 587386.746772550744936,  3488933.817566382698715,
         5290575.784156281501055},
       {-226426.343401445570635,  1035421.647801387240179,
        -6271459.446578867733479},
       {  39553.214744714961853, -1122384.858932408038527,
        -6257455.705907705239952},
       {-2197035.039946643635631,  4766296.481927301734686,
        -3612087.398071805480868},
       {2475217.167525716125965,  -590067.244431337225251,
         5836531.74855871964246 },
       {3251592.655810729600489,  2256703.30570419318974 ,
        -4985277.930962197482586},
       {-1850635.103680874686688, -3036577.247930331621319,
         5280569.380736761726439},
       { -95048.576977927994449,  -204957.529435861855745,
         6352981.530775795690715},
       {-106608.855637043248862,    47844.679874961388123,
        -6359984.3118050172925  },
       { 218676.696484291809611, -3026189.824885316658765,
         5592409.664520519785583}};

    for(int i=0; i<15;i++)
    {
        cout << endl << " [Point " << i+1 << " ]" << endl;
        {
            vector<double> rxyz(3), rllh(3);
            vector<double> xyz(3),llh(3);

            rllh.assign( ref_llh[i], ref_llh[i] + 3);
            rxyz.assign( ref_xyz[i], ref_xyz[i] + 3);

            llh = rllh;
            wgs84.latLonToXyz(xyz, llh);
            bool stat = checkAlmostEqual(xyz, rxyz, 9);
            cout << " [LLH_2_XYZ] ";
            if (stat) cout << "PASSED";
            cout << endl;

            xyz = rxyz;
            wgs84.xyzToLatLon(xyz, llh);
            stat = checkAlmostEqual(llh, rllh,9);
            cout << " [XYZ_2_LLH] ";
            if(stat) cout << "PASSED";
            cout << endl;
        }
    }
	

}

int main(int argc, char **argv) {
    /*
     * Ellipsoid unit-testing script.
     */

    testCorners();
    testCoords();

    return 0;
}
