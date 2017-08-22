//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_PEGTRANS_H__
#define __ISCE_CORE_PEGTRANS_H__

#include <vector>
#include "isce/core/Ellipsoid.h"
#include "isce/core/Peg.h"

namespace isce::core {
    struct Pegtrans {
        std::vector<std::vector<double>> mat;
        std::vector<std::vector<double>> matinv;
        std::vector<double> ov;
        double radcur;
    
        Pegtrans(double rd) : mat(3,std::vector<double>(3,0.)), matinv(3,std::vector<double>(3,0.)), ov(3,0.), radcur(rd) {}    // Value constructor
        Pegtrans() : Pegtrans(0.) {}                                                                                            // Default constructor (delegated)
        Pegtrans(const Pegtrans &p) : mat(p.mat), matinv(p.matinv), ov(p.ov), radcur(p.radcur) {}                               // Copy constructor
        inline Pegtrans& operator=(const Pegtrans&);
        
        void radarToXYZ(Ellipsoid&,Peg&);
        void convertSCHtoXYZ(std::vector<double>&,std::vector<double>&,int);
        void convertSCHdotToXYZdot(std::vector<double>&,std::vector<double>&,std::vector<double>&,std::vector<double>&,int);
        void SCHbasis(std::vector<double>&,std::vector<std::vector<double>>&,std::vector<std::vector<double>>&);
    };

    inline Pegtrans& Pegtrans::operator=(const Pegtrans &rhs) {
        mat = rhs.mat;
        matinv = rhs.matinv;
        ov = rhs.ov;
        radcur = rhs.radcur;
        return *this;
    }
}

#endif
