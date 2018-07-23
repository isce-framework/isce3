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
        
        void radarToXYZ(const Ellipsoid &, const Peg &);

        void convertXYZtoSCH(const cartesian_t & xyzv, cartesian_t & schv) const;
        void convertSCHtoXYZ(const cartesian_t & schv, cartesian_t & xyzv) const;
        void convertXYZdotToSCHdot(const cartesian_t & sch, const cartesian_t & xyzdot,
                                   cartesian_t & schdot) const;
        void convertSCHdotToXYZdot(const cartesian_t & sch, const cartesian_t & schdot,
                                   cartesian_t & xyzdot) const;
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
