//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_PEGTRANS_H__
#define __ISCE_CORE_PEGTRANS_H__

#include <vector>
#include "Constants.h"
#include "Ellipsoid.h"
#include "DenseMatrix.h"
#include "Peg.h"

namespace isce { namespace core {

    /** Data structure to assist with Peg point transformations
     *
     * This data structure stores matrices and offset vectors needed
     * to transform ECEF vectors to SCH and vice-versa */
    struct Pegtrans {

        /** Transformation matrix from SCH to ECEF*/
        cartmat_t mat;

        /** Transformation matrix from ECEF to SCH*/
        cartmat_t matinv;

        /** Offset vector between center of Ellipsoid and center of local sphere*/
        cartesian_t ov;

        /** Radius of curvature of local sphere*/
        double radcur;

        /** Empty constructor */
        Pegtrans() {}

        /** Copy constructor */
        Pegtrans(const Pegtrans &p) : mat(p.mat), matinv(p.matinv), ov(p.ov), radcur(p.radcur) {}

        /** Compute transformation matrices for a given Peg point
         *
         * @param[in] elp Ellipsoid object
         * @param[in] p Peg object*/
        void radarToXYZ(const Ellipsoid &elp, const Peg &p);

        /** Transform ECEF coordinates to SCH
         *
         * @param[in] xyzv ECEF position (m)
         * @param[out] schv SCH position (m)*/
        void convertXYZtoSCH(const cartesian_t & xyzv, cartesian_t & schv) const;

        /** Transform SCH coordinates to ECEF
         *
         * @param[in] schv SCH position (m)
         * @param[out] xyzv ECEF position (m)*/
        void convertSCHtoXYZ(const cartesian_t & schv, cartesian_t & xyzv) const;

        /** Transform ECEF velocity to SCH
         *
         * @param[in] sch SCH coordinates of platform from convertXYZtoSCH
         * @param[in] xyzdot ECEF velocity in m/s
         * @param[out] schdot SCH velocity in m/s*/
        void convertXYZdotToSCHdot(const cartesian_t & sch, const cartesian_t & xyzdot,
                                   cartesian_t & schdot) const;

        /** Transform SCH velocity to ECEF
         *
         * @param[in] sch SCH coordinates 
         * @param[in] schdot SCH velocity in m/s
         * @param[out] xyzdot ECEF velocity in m/s*/
        void convertSCHdotToXYZdot(const cartesian_t & sch, const cartesian_t & schdot,
                                   cartesian_t & xyzdot) const;

        /** Compute the transform matrix from ECEF to local SCH frame*/ 
        void SCHbasis(const cartesian_t &,cartmat_t&,cartmat_t&) const;
    };
}}

#endif
