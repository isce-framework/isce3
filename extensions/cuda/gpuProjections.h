/**
 * Source Author: Paulo Penteado, based on Projections.h by Piyush Agram / Joshua Cohen
 * Copyright 2018
 */

#ifndef __ISCE_CUDA_CORE_PROJECTIONS_H__
#define __ISCE_CUDA_CORE_PROJECTIONS_H__

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#define CUDA_GLOBAL
#endif

#include <cmath>
#include <iostream>
#include <vector>
#include "Constants.h"
#include "gpuEllipsoid.h"
#include "Projections.h"
using isce::core::cartesian_t;
using isce::core::Ellipsoid;
using isce::cuda::core::gpuEllipsoid;


namespace isce { namespace cuda { namespace core {




    
    /** Abstract base class for individual projections
     *
     *Internally, every derived class is expected to provide two functions.
     * forward - To convert llh (radians) to expected projection system 
     * inverse - To convert expected projection system to llh (radians)
     */
    struct ProjectionBase {
    	
    	
        /** Ellipsoid object for projections - currently only WGS84 */
        gpuEllipsoid ellipse;
        /** Type of projection system. This can be used to check if projection systems are equal
         * Private member and should not be modified after initialization*/
        int _epsgcode;

        /** Value constructor with EPSG code as input. Ellipsoid is always initialized to standard WGS84 ellipse.*/
        ProjectionBase(int code) : ellipse(6378137.,.0066943799901), _epsgcode(code) {}
        //__host__ __device__ ProjectionBase() : ellipse(6378137.,.0066943799901), _epsgcode(0) {}
        ProjectionBase() : ProjectionBase(0) {}



        /** Print function for debugging */
        virtual void print() const = 0;

        /** \brief Function for transforming from LLH. This is similar to fwd or fwd3d in PROJ.4 
         * 
         * @param[in] llh Lon/Lat/Height - Lon and Lat are in radians
         * @param[out] xyz Coordinates in specified projection system */
        CUDA_HOST virtual int forward(const cartesian_t& llh, cartesian_t& xyz) const = 0 ;
        CUDA_HOSTDEV virtual int forward(const double llh[], double xyz[]) const = 0 ;

        /** Function for transforming to LLH. This is similar to inv or inv3d in PROJ.4
         *
         * @param[in] xyz Coordinates in specified projection system 
         * @param[out] llh Lat/Lon/Height - Lon and Lat are in radians */
        CUDA_HOST virtual int inverse(const cartesian_t& xyz, cartesian_t& llh) const = 0 ;
        CUDA_HOSTDEV virtual int inverse(double xyz[], double llh[]) const = 0 ;
	    virtual double roundtriptest(int) ;
    };


    // Polar stereographic coordinate system
    struct PolarStereo  {
    	
        /** Ellipsoid object for projections - currently only WGS84 */
        gpuEllipsoid ellipse;
        /** Type of projection system. This can be used to check if projection systems are equal
         * Private member and should not be modified after initialization*/
        int _epsgcode;
    	
        // Constants related to projection system
        double lat0, lon0, lat_ts, akm1, e;
        bool isnorth;

        // Value constructor
        CUDA_HOSTDEV PolarStereo(int) ;
        CUDA_HOSTDEV PolarStereo() : PolarStereo(3413) {}

        inline void print() const;
        // Transfrom from LLH to Polar Stereo
        CUDA_HOST int forward(const cartesian_t&,cartesian_t&) const;
        CUDA_HOSTDEV int forward(double[], double[]) const;
        // Transform from Polar Stereo to LLH
        CUDA_HOST int inverse(const cartesian_t&,cartesian_t&) const;
        CUDA_HOSTDEV int inverse(double[], double[]) const;
        //Test round trip conversions on large arrays of points
        double roundtriptest(int) ;
    };
 
    inline void PolarStereo::print() const {
//        std::cout << "Projection: " << (isnorth ? "North" : "South") << " Polar Stereographic" <<
//                     std::endl << "EPSG: " << _epsgcode << std::endl;
    }


    // This is to transform a point from one coordinate system to another
    CUDA_HOSTDEV int projTransform(ProjectionBase* in, ProjectionBase *out, const cartesian_t &inpts,
                      cartesian_t &outpts);
}}}

#endif
