/**
 * Source Author: Paulo Penteado, based on Projections.h by Piyush Agram / Joshua Cohen
 * Copyright 2018
 */

#ifndef __ISCE_CUDA_CORE_PROJECTIONS_H__
#define __ISCE_CUDA_CORE_PROJECTIONS_H__

#include <cmath>
#include <iostream>
#include <vector>

#include <isce/core/Ellipsoid.h>
#include <isce/core/Projections.h>

namespace isce { namespace cuda { namespace core {

    /** Abstract base class for individual projections
     *
     *Internally, every derived class is expected to provide two functions.
     * forward - To convert llh (radians) to expected projection system 
     * inverse - To convert expected projection system to llh (radians)
     */
    struct ProjectionBase {
        typedef isce::core::Vec3        Vec3;

        /** Ellipsoid object for projections - currently only WGS84 */
        isce::core::Ellipsoid ellipse;
        /** Type of projection system. This can be used to check if projection systems are equal
         * Private member and should not be modified after initialization*/
        int _epsgcode;

        /** Value constructor with EPSG code as input. Ellipsoid is always initialized to standard WGS84 ellipse.*/
        CUDA_HOSTDEV ProjectionBase(int code) : ellipse(6378137.,.0066943799901), _epsgcode(code) {}

        /** \brief Host function for transforming from LLH. This is similar to fwd or fwd3d in PROJ.4 
         * 
         * @param[in] llh Lon/Lat/Height - Lon and Lat are in radians
         * @param[out] xyz Coordinates in specified projection system */
        CUDA_HOST int forward_h(const Vec3& llh, Vec3& xyz) const;

        /** \brief Device function for transform from LLH.
         *
         * @param[in] llh Lon/Lat/Height - Lon and Lat are in radians
         * @param[out] xyz Coordinates in specified coordinate system*/
        CUDA_DEV virtual int forward(const Vec3& llh, Vec3& xyz) const = 0;

        /** Host function for transforming to LLH. This is similar to inv or inv3d in PROJ.4
         *
         * @param[in] xyz Coordinates in specified projection system 
         * @param[out] llh Lon/Lat/Height - Lon and Lat are in radians */
        CUDA_HOST int inverse_h(const Vec3& xyz, Vec3& llh) const;

        /** Device function for tranforming to LLH.
         *
         * @param[in] xyz Coordinates in specified projection system
         * @param[out] llh Lon/Lat/Height - Lon and Lat are in radians */
        CUDA_DEV virtual int inverse(const Vec3& xyz, Vec3& llh) const = 0 ;
    };

    /** Geodetic Lon/Lat projection - EPSG:4326 */
    struct LonLat: public ProjectionBase {
        // Value Constructor
        CUDA_HOSTDEV LonLat() : ProjectionBase(4326) {}
        // Radians to Degrees pass through
        CUDA_DEV int forward(const Vec3&, Vec3&) const;
        // Degrees to Radians pass through
        CUDA_DEV int inverse(const Vec3&, Vec3&) const;
    };

    /** Standard WGS84 ECEF coordinates - EPSG:4978 */
    struct Geocent: public ProjectionBase {
        // Value Constructor
        CUDA_HOSTDEV Geocent():ProjectionBase(4978){};
        /** Same as Ellipsoid::lonLatToXyz */
        CUDA_DEV int forward(const Vec3&, Vec3&) const;

        /** Same as Ellipsoid::xyzTolonLat */
        CUDA_DEV int inverse(const Vec3&, Vec3&) const;
    };

    /** UTM coordinate extension of ProjectionBase
     *
     * EPSG 32601-32660 for Northern Hemisphere
     * EPSG 32701-32760 for Southern Hemisphere*/
    struct UTM : public ProjectionBase {
        // Constants related to the projection system
        double lon0;
        int zone;
        bool isnorth;
        // Parameters from Proj.4
        double cgb[6], cbg[6], utg[6], gtu[6];
        double Qn, Zb;

        // Value constructor
        CUDA_HOSTDEV UTM(int);

        /** Transform from llh (rad) to UTM (m)*/
        CUDA_DEV int forward(const Vec3&, Vec3&) const;

        /** Transform from UTM(m) to llh (rad)*/
        CUDA_DEV int inverse(const Vec3&, Vec3&) const;
    };
    
    /** Polar stereographic coordinate system
     *
     * EPSG:3413 - Greenland
     * EPSG:3031 - Antarctica */
    struct PolarStereo: public ProjectionBase {
    	
        // Constants related to projection system
        double lon0, lat_ts, akm1, e;
        bool isnorth;

        // Value constructor
        CUDA_HOSTDEV PolarStereo(int);
        /** Transfrom from LLH to Polar Stereo */
        CUDA_DEV int forward(const Vec3&, Vec3&) const;
        /** Transform from Polar Stereo to LLH */
        CUDA_DEV int inverse(const Vec3&, Vec3&) const;
    };

    /** Equal Area Projection extension of ProjBase
     *
     * EPSG:6933 for EASE2 grid*/
    struct CEA: public ProjectionBase {
        // Constants related to projection system
        double apa[3];
        double lat_ts, k0, e, one_es, qp;

        // Value constructor
        CUDA_HOSTDEV CEA();

        /** Transform from llh (rad) to CEA (m)*/
        CUDA_DEV int forward(const Vec3&, Vec3&) const;

        /** Transform from CEA (m) to LLH (rad)*/
        CUDA_DEV int inverse(const Vec3&, Vec3&) const;
    };
 
    // This is to transform a point from one coordinate system to another
    CUDA_DEV int projTransform(ProjectionBase* in, 
                               ProjectionBase *out,
                               const double *inpts,
                               double *outpts);

    //Projection Factory using EPSG code
    CUDA_HOSTDEV ProjectionBase* createProj(int epsg);

    CUDA_DEV int projInverse(int code, const double* in, double* out_llh);
    CUDA_DEV int projInverse(int code, const isce::core::Vec3& in, isce::core::Vec3& out_llh);
}}}

#endif
