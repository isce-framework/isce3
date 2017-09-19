//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_PEGTRANS_H__
#define __ISCE_CORE_PEGTRANS_H__

#include <vector>
#include "Constants.h"
#include "Ellipsoid.h"
#include "Peg.h"

namespace isce { namespace core {
    struct Pegtrans {
        std::vector<std::vector<double>> mat;
        std::vector<std::vector<double>> matinv;
        std::vector<double> ov;
        double radcur;
    
        Pegtrans(double rd) : mat(3,std::vector<double>(3,0.)), matinv(3,std::vector<double>(3,0.)), 
                              ov(3,0.), radcur(rd) {}
        Pegtrans() : Pegtrans(0.) {}
        Pegtrans(const Pegtrans &p) : mat(p.mat), matinv(p.matinv), ov(p.ov), radcur(p.radcur) {}
        inline Pegtrans& operator=(const Pegtrans&);
        
        void radarToXYZ(Ellipsoid&,Peg&);
        void convertSCHtoXYZ(std::vector<double>&,std::vector<double>&,orbitConvMethod);
        void convertSCHdotToXYZdot(std::vector<double>&,std::vector<double>&,std::vector<double>&,
                                   std::vector<double>&,orbitConvMethod);
        void SCHbasis(std::vector<double>&,std::vector<std::vector<double>>&,
                      std::vector<std::vector<double>>&);
    };

    inline Pegtrans& Pegtrans::operator=(const Pegtrans &rhs) {
        mat = rhs.mat;
        matinv = rhs.matinv;
        ov = rhs.ov;
        radcur = rhs.radcur;
        return *this;
    }
}}

#endif
