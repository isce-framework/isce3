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
        cartmat_t mat;
        cartmat_t matinv;
        cartesian_t ov;
        double radcur;
    
        Pegtrans(double rd) {}
        Pegtrans() : Pegtrans(0.) {}
        Pegtrans(const Pegtrans &p) : mat(p.mat), matinv(p.matinv), ov(p.ov), radcur(p.radcur) {}
        inline Pegtrans& operator=(const Pegtrans&);
        
        void radarToXYZ(const Ellipsoid&,const Peg&);
        void convertSCHtoXYZ(cartesian_t&,cartesian_t&,orbitConvMethod) const;
        void convertSCHdotToXYZdot(const cartesian_t&,const cartesian_t&,
                                   cartesian_t&,cartesian_t&,orbitConvMethod) const;
        void SCHbasis(const cartesian_t &,cartmat_t&,cartmat_t&) const;
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
