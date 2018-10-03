//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCE_CORE_POLY1D_H
#define ISCE_CORE_POLY1D_H

#include <stdexcept>
#include <string>
#include <vector>

// Declaration
namespace isce {
    namespace core {
        struct Poly1d;
    }
}

/** Data structure for representing 1D polynomials
 *
 * Poly1D is function of the form 
 * \f[
 *     f\left( x \right) = \sum_{i=0}^{N} a_i \cdot \left( \frac{x-\mu}{\sigma} \right)^i
 * \f]
 *
 * where \f$a_i\f$ represents the coefficients, \f$\mu\f$ represents the mean and 
 * \f$\sigma\f$ represents the norm*/
struct isce::core::Poly1d {
    /** Order of the polynomial */
    int order;
    /** Mean of the polynomial */
    double mean;
    /** Norm of the polynomial */
    double norm;
    /** Coefficients of the polynomial */
    std::vector<double> coeffs;

    /** Constructor 
     *
     * @param[in] ord Order
     * @param[in] mn Mean
     * @param[in] nm Norm */
    Poly1d(int ord, double mn, double nm) : order(ord), mean(mn), norm(nm), coeffs(ord+1) {}

    /** Empty constructor */
    Poly1d() : Poly1d(-1,0.,1.) {}

    /** Copy constructor */
    Poly1d(const Poly1d &p) : order(p.order), mean(p.mean), norm(p.norm), coeffs(p.coeffs) {}

    /** Assignment operator */
    inline Poly1d& operator=(const Poly1d&);

    /** Set coefficient by index
     *
     * @param[in] ind Index to set
     * @param[in] val Coefficient value*/
    inline void setCoeff(int ind,double val);

    /** Get coefficient by index 
     *
     * @param[in] ind Index to get*/
    inline double getCoeff(int ind) const;

    /** Evaluate polynomial at x
     *
     * @param[in] x Input x*/
    double eval(double x) const;

    /** Print for debugging */
    void printPoly() const;

    /** Return derivative of polynomial*/
    Poly1d derivative() const;
};

isce::core::Poly1d & isce::core::Poly1d::
operator=(const Poly1d &rhs) {
    order = rhs.order;
    mean = rhs.mean;
    norm = rhs.norm;
    coeffs = rhs.coeffs;
    return *this;
}

void isce::core::Poly1d::
setCoeff(int idx, double val) {
    if ((idx < 0) || (idx > order)){
        std::string errstr = "Poly1d::setCoeff - Trying to set coefficient " + 
                             std::to_string(idx+1) + " out of " + std::to_string(order+1);
        throw std::out_of_range(errstr);
    }
    coeffs[idx] = val;
}

double isce::core::Poly1d::
getCoeff(int idx) const {
    if ((idx < 0) || (idx > order)) {
        std::string errstr = "Poly1d::getCoeff - Trying to get coefficient " + 
                             std::to_string(idx+1) + " out of " + std::to_string(order+1);
        throw std::out_of_range(errstr);
    }
    return coeffs[idx];
}

#endif

// end of file
