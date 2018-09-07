//
// Source Author: Paulo Penteado, based on Projections.h by Piyush Agram / Joshua Cohen
// Copyright 2018
//
#ifndef __ISCE_CORE_CUDA_PROJECTIONS_H__
#define __ISCE_CORE_CUDA_PROJECTIONS_H__

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
#include "Ellipsoid.h" //To be replaced with gpuEllipsoid.h
#include "gpuEllipsoid.h"
#include "Projections.h"
using isce::core::cartesian_t;
using isce::core::Ellipsoid;
using isce::cuda::core::gpuEllipsoid;


namespace isce { 
    namespace cuda { 
        namespace core {

    /** Abstract base class for individual projections
     *
     *Internally, every derived class is expected to provide two functions.
     * forward - To convert llh (radians) to expected projection system 
     * inverse - To convert expected projection system to llh (radians) */
    struct ProjectionBase {
    	
        // TODO: make a copyconstructor from a CPU ProjectionBase
    	// TODO: create cpuproj
    	
    	//isce::core::ProjectionBase cpuproj;
        /** Ellipsoid object for projections - currently only WGS84 */
        gpuEllipsoid ellipse;
        /** Type of projection system. This can be used to check if projection systems are equal
         * Private member and should not be modified after initialization*/
        int _epsgcode;

        /** Value constructor with EPSG code as input. Ellipsoid is always initialized to standard WGS84 ellipse.*/
        CUDA_HOSTDEV ProjectionBase(int code) : ellipse(6378137.,.0066943799901), _epsgcode(code) {}

        /** Print function for debugging */
        virtual void print() const = 0;

        /** \brief Function for transforming from LLH. This is similar to fwd or fwd3d in PROJ.4 
         * 
         * @param[in] llh Lon/Lat/Height - Lon and Lat are in radians
         * @param[out] xyz Coordinates in specified projection system */
        CUDA_HOST int forward_h(const cartesian_t& llh, cartesian_t& xyz) const;
        CUDA_DEV virtual int forward(const double *llh, double *xyz) const = 0 ;

        /** Function for transforming to LLH. This is similar to inv or inv3d in PROJ.4
         *
         * @param[in] xyz Coordinates in specified projection system 
         * @param[out] llh Lat/Lon/Height - Lon and Lat are in radians */
        CUDA_HOST int inverse_h(const cartesian_t& xyz, cartesian_t& llh) const;
        CUDA_DEV virtual int inverse(const double *xyz, double *llh) const = 0 ;
	    
    };


    // Polar stereographic coordinate system
    struct PolarStereo : public ProjectionBase {
        // Constants related to projection system
        double lat0, lon0, lat_ts, akm1, e;
        bool isnorth;

        // Value constructor
        CUDA_HOSTDEV PolarStereo(int);

        inline void print() const;
        // Transfrom from LLH to Polar Stereo
        CUDA_DEV int forward(const double*, double*) const;
        // Transform from Polar Stereo to LLH
        CUDA_DEV int inverse(const double*, double*) const;
    };
 
    inline void PolarStereo::print() const {
        std::cout << "Projection: " << (isnorth ? "North" : "South") << " Polar Stereographic" <<
                     std::endl << "EPSG: " << _epsgcode << std::endl;
    }



    // This is to transform a point from one coordinate system to another
    CUDA_HOST int projTransform_h(const ProjectionBase* in, const ProjectionBase *out, const cartesian_t &inpts,
                      cartesian_t &outpts);
        }
    }
}

#endif
