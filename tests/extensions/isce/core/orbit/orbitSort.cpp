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

bool checkEqual(vector<double> &ref, vector<double> &calc) {
    /*
     *  Calculate if two vectors are almost equal to n_digits of precision.
     */

    bool stat = true;
    for (int i=0; i<static_cast<int>(ref.size()); i++) {
        stat = stat & (ref[i] == calc[i]);
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

void testReverse() {
    /*
     * Test linear orbit.
     */

    Orbit orb(1,11);
    double t = 1000.;
    double t1;
    vector<double> opos = {0., 0., 0.};
    vector<double> ovel = {4000., -1000., 4500.};
    vector<double> pos(3), vel(3);

    // Create straight-line orbit with 11 state vectors, each 10 s apart
    for (int i=0; i<11; i++) {
        makeLinearSV(i*10., opos, ovel, pos, vel);
        orb.setStateVector(i, t+(i*10.), pos, vel);
    }


    Orbit newOrb(1,0);

    for(int i=10; i>=0; i--)
    {
        orb.getStateVector(i, t, pos, vel);
        newOrb.addStateVector(t,pos,vel);
    }

    bool stat = true;

    cout << " [Add State Vector in reverse] " << endl;

    // Test each interpolation time against SCH, Hermite, and Legendre interpolation methods
    for (int i=0; i<10; i++) {
        orb.getStateVector(i, t, pos, vel);
        newOrb.getStateVector(i, t1, opos, ovel);

        stat = stat & (t == t1);
        stat = stat & checkEqual(pos, opos);
        stat = stat & checkEqual(vel, ovel);
        cout << " [index = " << i <<"] ";
        if (stat) cout << "PASSED";
        cout << endl;
    }

}

void testOutOfOrder() {
    /*
     * Test linear orbit.
     */

    Orbit orb(1,11);
    double t = 1000.;
    double t1;
    vector<double> opos = {0., 0., 0.};
    vector<double> ovel = {4000., -1000., 4500.};
    vector<double> pos(3), vel(3);

    // Create straight-line orbit with 11 state vectors, each 10 s apart
    for (int i=0; i<11; i++) {
        makeLinearSV(i*10., opos, ovel, pos, vel);
        orb.setStateVector(i, t+(i*10.), pos, vel);
    }


    Orbit newOrb(1,0);

    for(int i=10; i>=0; i-=2)
    {
        orb.getStateVector(i, t, pos, vel);
        newOrb.addStateVector(t,pos,vel);
    }


    for(int i=1; i<10; i+=2)
    {
        orb.getStateVector(i, t, pos, vel);
        newOrb.addStateVector(t, pos, vel);
    }

    bool stat = true;

    cout << " [Add State Vector out of order] " << endl;

    // Test each interpolation time against SCH, Hermite, and Legendre interpolation methods
    for (int i=0; i<10; i++) {
        orb.getStateVector(i, t, pos, vel);
        newOrb.getStateVector(i, t1, opos, ovel);

        stat = stat & (t == t1);
        stat = stat & checkEqual(pos, opos);
        stat = stat & checkEqual(vel, ovel);
        cout << " [index = " << i <<"] ";
        if (stat) cout << "PASSED";
        cout << endl;
    }

}


int main(int argc, char **argv) {
    /*
     * Orbit unit-testing script.
     */

    testReverse();
    testOutOfOrder();

    return 0;
}
