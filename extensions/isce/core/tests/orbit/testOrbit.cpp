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
    for (int i=0; i<static_cast<int>(ref.size()); i++) {
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

void testStraightLine() {
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
    double test_t[] = {23.3, 36.7, 54.5, 89.3};
    vector<double> ref_pos(3), ref_vel(3);

    cout << endl << "[Linear orbit interpolation]" << endl;

    // Test each interpolation time against SCH, Hermite, and Legendre interpolation methods
    bool stat = true;
    cout << " [SCH Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makeLinearSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        orb.interpolate(t+test_t[i], pos, vel, SCH_METHOD);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 4);
        if (stat) cout << "PASSED";
        cout << endl;
    }

    stat = true;
    cout << " [Hermite Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makeLinearSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        orb.interpolate(t+test_t[i], pos, vel, HERMITE_METHOD);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 4);
        if (stat) cout << "PASSED";
        cout << endl;
    }

    stat = true;
    cout << " [Legendre Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makeLinearSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        orb.interpolate(t+test_t[i], pos, vel, LEGENDRE_METHOD);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 4);
        if (stat) cout << "PASSED";
        cout << endl;
    }
}

void makeCircularSV(double dt, vector<double> &opos, vector<double> &ovel, vector<double> &pos,
                    vector<double> &vel) {
    double omega1 = (2. * M_PI) / 7000.;
    double omega2 = (2. * M_PI) / 4000.;
    double theta1 = (2. * M_PI) / 8.;
    double theta2 = (2. * M_PI) / 12.;
    double radius = 8000000.;
    double ang1 = theta1 + (dt * omega1);
    double ang2 = theta2 + (dt * omega2);
    pos = {opos[0] + (radius * cos(ang1)),
           opos[1] + (radius * (sin(ang1) + cos(ang2))),
           opos[2] + (radius * sin(ang2))};
    vel = {radius * -omega1 * sin(ang1),
           radius * ((omega1 * cos(ang1)) - (omega2 * sin(ang2))),
           radius * omega2 * cos(ang2)};
}

void testCircle() {
    /*
     * Test circular orbit.
     */

    Orbit orb(1,11);
    double t = 1000.;
    vector<double> opos = {7000000., -4500000., 7800000.};
    vector<double> ovel(3,0.), pos(3,0.), vel(3,0.);

    // Create circular orbit with 11 state vectors, each 5 s apart
    for (int i=0; i<11; i++) {
        makeCircularSV(i*5., opos, ovel, pos, vel);
        orb.setStateVector(i, t+(i*5.), pos, vel);
    }

    // Interpolation test times
    double test_t[] = {11.65, 18.35, 27.25, 44.65};
    vector<double> ref_pos(3), ref_vel(3);

    cout << endl << "[Circular orbit interpolation]" << endl;

    // Test each interpolation time against SCH, Hermite, and Legendre interpolation methods
    bool stat = true;
    cout << " [SCH Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makeCircularSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        orb.interpolate(t+test_t[i], pos, vel, SCH_METHOD);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 4);
        if (stat) cout << "PASSED";
        cout <<endl;
    }

    stat = true;
    cout << " [Hermite Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makeCircularSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        orb.interpolate(t+test_t[i], pos, vel, HERMITE_METHOD);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 4);
        if (stat) cout << "PASSED";
        cout <<endl;
    }

    stat = true;
    cout << " [Legendre Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makeCircularSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        orb.interpolate(t+test_t[i], pos, vel, LEGENDRE_METHOD);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 4);
        if (stat) cout << "PASSED";
        cout <<endl;
    }
}



void makePolynomialSV(double dt, vector<double> &xpoly, vector<double> &ypoly,
                                 vector<double> &zpoly, vector<double> &pos,
                                 vector<double> &vel) {

    pos[0] = 0.0;
    double fact = 1.0;
    for (int i=0; i < static_cast<int>(xpoly.size()); i++, fact*=dt)
    {
        pos[0] += fact * xpoly[i];
    }

    vel[0] = 0.0;
    fact = 1.0;
    for(int i=1; i < static_cast<int>(xpoly.size()); i++, fact*=dt)
    {
        vel[0] += i * xpoly[i] * fact;
    }


    pos[1] = 0.0;
    fact = 1.0;
    for (int i=0; i < static_cast<int>(ypoly.size()); i++, fact*=dt)
    {
        pos[1] += fact * ypoly[i];
    }

    vel[1] = 0.0;
    fact = 1.0;
    for(int i=1; i < static_cast<int>(ypoly.size()); i++, fact*=dt)
    {
        vel[1] += i * ypoly[i] * fact;
    }


    pos[2] = 0.0;
    fact = 1.0;
    for (int i=0; i < static_cast<int>(zpoly.size()); i++, fact*=dt)
    {
        pos[2] += fact * zpoly[i];
    }

    vel[2] = 0.0;
    fact = 1.0;
    for(int i=1; i < static_cast<int>(zpoly.size()); i++, fact*=dt)
    {
        vel[2] += i * zpoly[i] * fact;
    }

}

void testPolynomial() {
    /*
     * Test linear orbit.
     */

    Orbit orb(1,11);
    double t = 1000.;
    vector<double> pos(3), vel(3);

    vector<double> xpoly = {-7000000., 5435., -45.0, 7.3};
    vector<double> ypoly = {5400000., -4257., 23.0, 3.9, 0.01};
    vector<double> zpoly = {0.0, 7000., 11.0};

    // Create straight-line orbit with 11 state vectors, each 10 s apart
    for (int i=0; i<11; i++) {
        makePolynomialSV(i*10., xpoly, ypoly, zpoly, pos, vel);
        orb.setStateVector(i, t+(i*10.), pos, vel);
    }

    // Interpolation test times
    double test_t[] = {23.3, 36.7, 54.5, 89.3};
    vector<double> ref_pos(3), ref_vel(3);

    cout << endl << "[Polynomial orbit interpolation]" << endl;

    // Test each interpolation time against SCH, Hermite, and Legendre interpolation methods
    bool stat = true;
    cout << " [SCH Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makePolynomialSV(test_t[i], xpoly, ypoly, zpoly, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        orb.interpolate(t+test_t[i], pos, vel, SCH_METHOD);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 4);
        if (stat) cout << "PASSED";
        cout << endl;
    }

    stat = true;
    cout << " [Hermite Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makePolynomialSV(test_t[i], xpoly, ypoly, zpoly, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        orb.interpolate(t+test_t[i], pos, vel, HERMITE_METHOD);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 4);
        if (stat) cout << "PASSED";
        cout << endl;
    }

    stat = true;
    cout << " [Legendre Interpolation]" << endl;
    for (int i=0; i<4; i++) {
        makePolynomialSV(test_t[i], xpoly, ypoly, zpoly, ref_pos, ref_vel);
        cout << "  [t = " << test_t[i] << "] ";
        orb.interpolate(t+test_t[i], pos, vel, LEGENDRE_METHOD);
        stat = stat & checkAlmostEqual(ref_pos, pos, 3);
        stat = stat & checkAlmostEqual(ref_vel, vel, 4);
        if (stat) cout << "PASSED";
        cout << endl;
    }
}


int main(int argc, char **argv) {
    /*
     * Orbit unit-testing script.
     */

    testStraightLine();
    testCircle();
    testPolynomial();

    return 0;
}
