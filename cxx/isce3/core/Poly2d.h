#pragma once

#include "forward.h"

#include <stdexcept>
#include <string>
#include <vector>
#include "Constants.h"

/** Data structure for representing 2D polynomials
 *
 * Poly2D is function of the form
 * \f[
 *     f\left( y, x \right) = \sum_{i=0}^{N_y} \sum_{j=0}^{N_x} a_{ij} \cdot \left( \frac{y-\mu_y}{\sigma_y} \right)^i
 \cdot \left( \frac{x-\mu_x}{\sigma_x} \right)^j
 * \f]
 *
 * where \f$a_ij\f$ represents the coefficients, \f$\mu_x\f$ and \f$\mu_y\f$ represent the means and
 * \f$\sigma_x\f$ and \f$\sigma_y\f$ represent the norms*/
class isce3::core::Poly2d {
public:

    /** Order of polynomial in range or x*/
    int xOrder;
    /** Order of polynomial in azimuth or y*/
    int yOrder;
    /** Mean in range or x direction*/
    double xMean;
    /** Mean in azimuth or y direction*/
    double yMean;
    /** Norm in range or x direction*/
    double xNorm;
    /** Norm in azimuth or y direction*/
    double yNorm;
    /**Linearized vector of coefficients in row-major format*/
    std::vector<double> coeffs;

    /** Simple constructor
     *
     * @param[in] xo x/Range Order
     * @param[in] yo y/Azimuth Order
     * @param[in] xm x/Range Mean
     * @param[in] ym y/Azimuth Mean
     * @param[in] xn x/Range Norm
     * @param[in] yn y/Azimuth Norm*/
    Poly2d(int xo, int yo, double xm, double ym, double xn, double yn) : xOrder(xo),
                                                                         yOrder(yo),
                                                                         xMean(xm),
                                                                         yMean(ym),
                                                                         xNorm(xn),
                                                                         yNorm(yn),
                                                                         coeffs((xo+1)*(yo+1))
                                                                         {}

    /** Empty constructor*/
    Poly2d() : Poly2d(-1,-1,0.,0.,1.,1.) {}

    /** Copy constructor
     *
     * @param[in] p Poly2D object*/
    Poly2d(const Poly2d &p) : xOrder(p.xOrder), yOrder(p.yOrder),
                              xMean(p.xMean), yMean(p.yMean),
                              xNorm(p.xNorm), yNorm(p.yNorm),
                              coeffs(p.coeffs) {}

    /** Assignment operator*/
    inline Poly2d& operator=(const Poly2d&);

    /** Set coefficient by indices*/
    inline void setCoeff(int row, int col, double val);

    /**Get coefficient by indices*/
    inline double getCoeff(int row, int col) const;

    /**Evaluate polynomial at given y/azimuth/row ,x/range/col*/
    double eval(double y, double x) const;

    /**Printing for debugging*/
    void printPoly() const;
};

isce3::core::Poly2d & isce3::core::Poly2d::
operator=(const Poly2d &rhs) {
    xOrder = rhs.xOrder;
    yOrder = rhs.yOrder;
    xMean = rhs.xMean;
    yMean = rhs.yMean;
    xNorm = rhs.xNorm;
    yNorm = rhs.yNorm;
    coeffs = rhs.coeffs;
    return *this;
}

/**
 * @param[in] row azimuth/y index
 * @param[in] col range/x index
 * @param[in] val Coefficient value*/
void isce3::core::Poly2d::
setCoeff(int row, int col, double val) {
    if ((row < 0) || (row > yOrder)) {
        std::string errstr = "Poly2d::setCoeff - Trying to set coefficient for row " +
                             std::to_string(row+1) + " out of " +
                             std::to_string(yOrder+1);
        throw std::out_of_range(errstr);
    }
    if ((col < 0) || (col > xOrder)) {
        std::string errstr = "Poly2d::setCoeff - Trying to set coefficient for col " +
                             std::to_string(col+1) + " out of " + std::to_string(xOrder+1);
        throw std::out_of_range(errstr);
    }
    coeffs[IDX1D(row,col,xOrder+1)] = val;
}

/**
 * @param[in] row azimuth/y index
 * @param[in] col range/x index*/
double isce3::core::Poly2d::
getCoeff(int row, int col) const {
    if ((row < 0) || (row > yOrder)) {
        std::string errstr = "Poly2d::getCoeff - Trying to get coefficient for row " +
                             std::to_string(row+1) + " out of " +
                             std::to_string(yOrder+1);
        throw std::out_of_range(errstr);
    }
    if ((col < 0) || (col > xOrder)) {
        std::string errstr = "Poly2d::getCoeff - Trying to get coefficient for col " +
                             std::to_string(col+1) + " out of " + std::to_string(xOrder+1);
        throw std::out_of_range(errstr);
    }
    return coeffs[IDX1D(row,col,xOrder+1)];
}
