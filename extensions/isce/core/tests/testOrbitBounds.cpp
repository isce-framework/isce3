//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <iostream>
#include <vector>
#include "isce/core/Constants.h"
#include "isce/core/Orbit.h"
using isce::core::orbitInterpMethod;
using isce::core::HERMITE_METHOD;
using isce::core::LEGENDRE_METHOD;
using isce::core::SCH_METHOD;
using isce::core::Orbit;
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



void makeLinearSV(double dt, vector<double> &opos, vector<double> &ovel, vector<double> &pos,
                  vector<double> &vel) {
    pos = {opos[0] + (dt * ovel[0]), opos[1] + (dt * ovel[1]), opos[2] + (dt * ovel[2])};
    vel = ovel;
}

void testOutOfBounds() {
    /*
     * Test linear orbit.
     */

    Orbit orb(1,11);
    double t = 1000.;
    vector<double> opos = {0., 0., 0.};
    vector<double> ovel = {4000., -1000., 4500.};
    vector<double> pos(3), vel(3);

    // Create straight-line orbit with 11 state vectors, each 10 s apart
    for (int i=0; i<11; i++) {
        makeLinearSV(i*10., opos, ovel, pos, vel);
        orb.setStateVector(i, t+(i*10.), pos, vel);
    }
    
    // Interpolation test times
    double test_t[] = {-23.0, -1.0, 101.0, 112.0};
    vector<double> ref_pos(3), ref_vel(3);

    cout << endl << "[Out of bounds cases for orbit interpolation]" << endl;

    // Test each interpolation time against SCH, Hermite, and Legendre interpolation methods
    bool stat = true;
    cout << " [SCH Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makeLinearSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        stat = (orb.interpolate(t+test_t[i], pos, vel, SCH_METHOD) == 1);
        if (stat) cout << "PASSED";
        cout << endl;
    }

    stat = true;
    cout << " [Hermite Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makeLinearSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        stat = (orb.interpolate(t+test_t[i], pos, vel, HERMITE_METHOD) == 1);
        if (stat) cout << "PASSED";
        cout << endl;
    }

    stat = true;
    cout << " [Legendre Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makeLinearSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        stat = (orb.interpolate(t+test_t[i], pos, vel, LEGENDRE_METHOD) == 1);
        if (stat) cout << "PASSED";
        cout << endl;
    }
}


void testEdges() {
    /*
     * Test linear orbit.
     */

    Orbit orb(1,11);
    double t = 1000.;
    vector<double> opos = {0., 0., 0.};
    vector<double> ovel = {4000., -1000., 4500.};
    vector<double> pos(3), vel(3);

    // Create straight-line orbit with 11 state vectors, each 10 s apart
    for (int i=0; i<11; i++) {
        makeLinearSV(i*10., opos, ovel, pos, vel);
        orb.setStateVector(i, t+(i*10.), pos, vel);
    }
    
    // Interpolation test times
    double test_t[] = {0.0, 100.0};
    vector<double> ref_pos(3), ref_vel(3);

    cout << endl << "[Edge case for orbit interpolation]" << endl;

    // Test each interpolation time against SCH, Hermite, and Legendre interpolation methods
    bool stat = true;
    cout << " [SCH Interpolation]" << endl;
    for (int i=0; i<2; i++) {
        makeLinearSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        stat = (orb.interpolate(t+test_t[i], pos, vel, SCH_METHOD) == 0);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 3);
        if (stat) cout << "PASSED";
        cout << endl;
    }

    stat = true;
    cout << " [Hermite Interpolation]" << endl;
    for (int i=0; i<2; i++) {
        makeLinearSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        stat = (orb.interpolate(t+test_t[i], pos, vel, HERMITE_METHOD) == 0);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 3);
        if (stat) cout << "PASSED";
        cout << endl;
    }

    stat = true;
    cout << " [Legendre Interpolation]" << endl;
    for (int i=0; i<2; i++) {
        makeLinearSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        stat = (orb.interpolate(t+test_t[i], pos, vel, LEGENDRE_METHOD) == 0);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 3);
        if (stat) cout << "PASSED";
        cout << endl;
    }
}



int main(int argc, char **argv) {
    /*
     * Orbit unit-testing script.
     */

    testOutOfBounds();
    testEdges();

    return 0;
}
