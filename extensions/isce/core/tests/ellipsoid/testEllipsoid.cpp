//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <iostream>
#include <vector>
#include "isce/core/Constants.h"
#include "isce/core/Ellipsoid.h"
using isce::core::latLonConvMethod;
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
        wgs84.latLon(xyz, llh, LLH_2_XYZ);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 7);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.latLon(xyz, llh, XYZ_2_LLH);
        stat = checkAlmostEqual(llh, ref_llh,7);
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
        wgs84.latLon(xyz, llh, LLH_2_XYZ);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 7);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.latLon(xyz, llh, XYZ_2_LLH);
        stat = checkAlmostEqual(llh, ref_llh,7);
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
        wgs84.latLon(xyz, llh, LLH_2_XYZ);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 7);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.latLon(xyz, llh, XYZ_2_LLH);
        stat = checkAlmostEqual(llh, ref_llh,7);
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
        wgs84.latLon(xyz, llh, LLH_2_XYZ);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 7);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.latLon(xyz, llh, XYZ_2_LLH);
        stat = checkAlmostEqual(llh, ref_llh,7);
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
        wgs84.latLon(xyz, llh, LLH_2_XYZ);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 7);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.latLon(xyz, llh, XYZ_2_LLH);
        stat = checkAlmostEqual(llh, ref_llh,7);
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
        wgs84.latLon(xyz, llh, LLH_2_XYZ);
        bool stat = checkAlmostEqual(xyz, ref_xyz, 7);
        cout << " [LLH_2_XYZ] ";
        if (stat) cout << "PASSED";
        cout << endl;

        xyz = ref_xyz;
        wgs84.latLon(xyz, llh, XYZ_2_LLH);
        stat = checkAlmostEqual(llh, ref_llh,7);
        cout << " [XYZ_2_LLH] ";
        if(stat) cout << "PASSED";
        cout << endl;
    }

}

int main(int argc, char **argv) {
    /*
     * Orbit unit-testing script.
     */

    testCorners();

    return 0;
}
