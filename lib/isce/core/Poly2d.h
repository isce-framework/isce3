//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCE_CORE_POLY2D_H
#define ISCE_CORE_POLY2D_H
#pragma once

#include "forward.h"

#include <stdexcept>
#include <string>
#include <vector>
#include "Constants.h"

/** Data structure for representing 1D polynomials
 *
 * Poly2D is function of the form 
 * \f[
 *     f\left( y, x \right) = \sum_{i=0}^{N_y} \sum_{j=0}^{N_x} a_{ij} \cdot \left( \frac{y-\mu_y}{\sigma_y} \right)^i 
 \cdot \left( \frac{x-\mu_x}{\sigma_x} \right)^j 
 * \f]
 *
 * where \f$a_ij\f$ represents the coefficients, \f$\mu_x\f$ and \f$\mu_y\f$ represent the means and 
 * \f$\sigma_x\f$ and \f$\sigma_y\f$ represent the norms*/
class isce::core::Poly2d {
public:

    /** Order of polynomial in range or x*/
    int rangeOrder;
    /** Order of polynomial in azimuth or y*/
    int azimuthOrder;
    /** Mean in range or x direction*/
    double rangeMean;
    /** Mean in azimuth or y direction*/
    double azimuthMean;
    /** Norm in range or x direction*/
    double rangeNorm;
    /** Norm in azimuth or y direction*/
    double azimuthNorm;
    /**Linearized vector of coefficients in row-major format*/
    std::vector<double> coeffs;

    /** Simple constructor
     *
     * @param[in] ro Range Order
     * @param[in] ao Azimuth Order
     * @param[in] rm Range Mean
     * @param[in] am Azimuth Mean
     * @param[in] rn Range Norm
     * @param[in] an Azimuth Norm*/
    Poly2d(int ro, int ao, double rm, double am, double rn, double an) : rangeOrder(ro), 
                                                                         azimuthOrder(ao), 
                                                                         rangeMean(rm), 
                                                                         azimuthMean(am),
                                                                         rangeNorm(rn), 
                                                                         azimuthNorm(an), 
                                                                         coeffs((ro+1)*(ao+1)) 
                                                                         {}

    /** Empty constructor*/
    Poly2d() : Poly2d(-1,-1,0.,0.,1.,1.) {}

    /** Copy constructor 
     *
     * @param[in] p Poly2D object*/
    Poly2d(const Poly2d &p) : rangeOrder(p.rangeOrder), azimuthOrder(p.azimuthOrder), 
                              rangeMean(p.rangeMean), azimuthMean(p.azimuthMean), 
                              rangeNorm(p.rangeNorm), azimuthNorm(p.azimuthNorm), 
                              coeffs(p.coeffs) {}

    /** Assignment operator*/
    inline Poly2d& operator=(const Poly2d&);

    /** Set coefficient by indices*/
    inline void setCoeff(int row, int col, double val);

    /**Get coefficient by indices*/
    inline double getCoeff(int row, int col) const;

    /**Evaluate polynomial at given y,x*/
    double eval(double azi, double rng) const;

    /**Printing for debugging*/
    void printPoly() const;
};

isce::core::Poly2d & isce::core::Poly2d::
operator=(const Poly2d &rhs) {
    rangeOrder = rhs.rangeOrder;
    azimuthOrder = rhs.azimuthOrder;
    rangeMean = rhs.rangeMean;
    azimuthMean = rhs.azimuthMean;
    rangeNorm = rhs.rangeNorm;
    azimuthNorm = rhs.azimuthNorm;
    coeffs = rhs.coeffs;
    return *this;
}

/**
 * @param[in] row azimuth/y index 
 * @param[in] col range/x index
 * @param[in] val Coefficient value*/
void isce::core::Poly2d::
setCoeff(int row, int col, double val) {
    if ((row < 0) || (row > azimuthOrder)) {
        std::string errstr = "Poly2d::setCoeff - Trying to set coefficient for row " + 
                             std::to_string(row+1) + " out of " + 
                             std::to_string(azimuthOrder+1);
        throw std::out_of_range(errstr);
    }
    if ((col < 0) || (col > rangeOrder)) {
        std::string errstr = "Poly2d::setCoeff - Trying to set coefficient for col " +
                             std::to_string(col+1) + " out of " + std::to_string(rangeOrder+1);
        throw std::out_of_range(errstr);
    }
    coeffs[IDX1D(row,col,rangeOrder+1)] = val;
}

/**
 * @param[in] row azimuth/y index
 * @param[in] col range/x index*/
double isce::core::Poly2d::
getCoeff(int row, int col) const {
    if ((row < 0) || (row > azimuthOrder)) {
        std::string errstr = "Poly2d::getCoeff - Trying to get coefficient for row " +
                             std::to_string(row+1) + " out of " + 
                             std::to_string(azimuthOrder+1);
        throw std::out_of_range(errstr);
    }
    if ((col < 0) || (col > rangeOrder)) {
        std::string errstr = "Poly2d::getCoeff - Trying to get coefficient for col " + 
                             std::to_string(col+1) + " out of " + std::to_string(rangeOrder+1);
        throw std::out_of_range(errstr);
    }
    return coeffs[IDX1D(row,col,rangeOrder+1)];
}

#endif

// end of file
