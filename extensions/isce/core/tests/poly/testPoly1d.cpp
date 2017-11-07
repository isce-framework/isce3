//-*- C++ -*-
//-*- coding: utf-8 -*-


#include <iostream>
#include <cmath>
#include <vector>
#include "isce/core/Poly1d.h"

void testConstant() {

    const double refval = 10.0;

    // Interpolate N values in x and y
    bool stat = true;
    for (size_t i = 1; i < 5; ++i) 
    {
        //Mean and norm should not matter
        isce::core::Poly1d poly(0, i*1.0, i*i*1.0);
        poly.setCoeff(0, refval);

        double value = poly.eval(i*1.0);
        stat = stat & (std::abs(value - refval) == 0);
    }

    std::cout << "\n[Poly1d constant] ";
    if (stat) 
        std::cout << "PASSED";
    else
        std::cout << "FAILED";
    std::cout << std::endl;

}


void testMeanShift()
{
    //Use identity polynomial for testing
    isce::core::Poly1d refpoly(2, 0.0, 1.0);
    refpoly.setCoeff(0, 0.0);
    refpoly.setCoeff(1, 1.0);
    refpoly.setCoeff(0, 0.0);

    bool stat = true;
    for(size_t i=0; i<5; i++)
    {
        isce::core::Poly1d newpoly(refpoly);
        newpoly.mean = 0.5 * i * i;

        double refval = refpoly.eval(2.0 * i);
        double newval = newpoly.eval(2.0 * i + 0.5 * i * i); 
        stat = stat & (std::abs(newval - refval) == 0);
    }

    std::cout << "\n[Poly1d meanshift] ";
    if (stat)
        std::cout << "PASSED";
    else
        std::cout << "FAILED";
    std::cout << std::endl;

}


void testNormShift()
{
    //Use square polynomial for testing
    isce::core::Poly1d refpoly(2,0.0,1.0);
    refpoly.setCoeff(0, 0.0);
    refpoly.setCoeff(1, 0.0);
    refpoly.setCoeff(2, 1.0);

    bool stat=true;
    for(size_t i=1; i<6; i++)
    {
        isce::core::Poly1d newpoly(refpoly);
        newpoly.norm = i * i * 1.0;

        double refval = refpoly.eval(2.5);
        double newval = newpoly.eval(2.5 * i * i);

        stat = stat & (std::abs(newval - refval) == 0);
    }

    std::cout << "\n[Poly1d normshift] ";
    if (stat)
        std::cout << "PASSED";
    else
        std::cout << "FAILED";
    std::cout << std::endl;

}

void testDerivative()
{
    //Use square polynomial for testing
    isce::core::Poly1d refpoly(5,0.0,1.0);


    bool stat=true;
    for(size_t i=1; i<6; i++)
    {
        isce::core::Poly1d refpoly(i, 0.0, 1.0);
        refpoly.norm = i;
        refpoly.setCoeff(0, 10.0);
        for (int ii=1; ii<=i; ii++)
        {
            refpoly.setCoeff(ii, 1.0/ii);
        }


        isce::core::Poly1d refder(i-1,0.0,1.0);
        refder.norm = i;
        std::fill( refder.coeffs.begin(), refder.coeffs.end(), 1.0/i);

        isce::core::Poly1d newpoly = refpoly.derivative();

        double refval = refder.eval(0.8);
        double newval = newpoly.eval(0.8);

        stat = stat & (std::abs(newval - refval) == 0);
    }

    std::cout << "\n[Poly1d derivative] ";
    if (stat)
        std::cout << "PASSED";
    else
        std::cout << "FAILED";
    std::cout << std::endl;

}



int main() {

    testConstant();
    testMeanShift();
    testNormShift();
    testDerivative();
    return 0;
}

// end of file
