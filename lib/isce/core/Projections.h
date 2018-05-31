//
// Source Author: Piyush Agram
// Co-Author: Joshua Cohen
// Copyright 2017
//
#ifndef __ISCE_CORE_PROJECTIONS_H__
#define __ISCE_CORE_PROJECTIONS_H__

#include <cmath>
#include <iostream>
#include <vector>
#include "Constants.h"
#include "Ellipsoid.h"

namespace isce { namespace core {

    // Abstract base class for individual projections
    struct ProjectionBase {
        // Ellipsoid to be used for all transformations
        Ellipsoid ellipse;
        // Type of projection system. This can be used to check if projection systems are equal
        // Private member and should not be modified after initialization
        int _epsgcode;

        // Value constructor with EPSG code as input. Ellipsoid is always initialized to standard
        // WGS84 ellipse.
        ProjectionBase(int code) : ellipse(6378137.,.0066943799901), _epsgcode(code) {}

        // Print function for debugging
        virtual void print() const = 0;
        // Function for transforming from LLH. This is similar to fwd or fwd3d in PROJ.4
        virtual int forward(const cartesian_t&,cartesian_t&) const = 0 ;
        // Function for transforming to LLH. This is similar to inv or inv3d in PROJ.4
        virtual int inverse(const cartesian_t&,cartesian_t&) const = 0 ;
    };

    // Standard WGS84 Lon/Lat Projection 
    struct LonLat : public ProjectionBase {
        // Value constructor
        LonLat() : ProjectionBase(4326) {}
        
        inline void print() const;
        // This will be a pass through for Lat/Lon
        inline int forward(const cartesian_t&,cartesian_t&) const;
        // This will also be a pass through for Lat/Lon
        inline int inverse(const cartesian_t&,cartesian_t&) const;
    };

    inline void LonLat::print() const {
        std::cout << "Projection: LatLon" << std::endl << "EPSG: " << _epsgcode << std::endl;
    }

    inline int LonLat::forward(const cartesian_t &in, cartesian_t &out) const {
        out = in;
        return 0;
    }

    inline int LonLat::inverse(const cartesian_t &in, cartesian_t &out) const {
        out = in;
        return 0;
    }

    // Standard WGS84 ECEF coordinates
    struct Geocent : public ProjectionBase {
        // Value constructor
        Geocent() : ProjectionBase(4978) {}
        
        inline void print() const;
        // This is same as latLonToXyz
        int forward(const cartesian_t&,cartesian_t&) const;
        // This is same as xyzToLatLon
        int inverse(const cartesian_t&,cartesian_t&) const;
    };

    inline void Geocent::print() const {
        std::cout << "Projection: Geocent" << std::endl << "EPSG: " << _epsgcode << std::endl;
    }

    // UTM coordinates
    struct UTM : public ProjectionBase { 
        // Constants related to the projection system
        double lon0;
        int zone;
        bool isnorth;
        // Parameters from Proj.4
        double cgb[6], cbg[6], utg[6], gtu[6];
        double Qn, Zb;

        // Value constructor
        UTM(int);

        inline void print() const;
        // Transform from LLH to UTM
        int forward(const cartesian_t&,cartesian_t&) const;
        // Transform from UTM to LLH
        int inverse(const cartesian_t&,cartesian_t&) const;
    };

    inline void UTM::print() const {
        std::cout << "Projection: UTM" << std::endl << "Zone: " << zone << (isnorth ? "N" : "S") << 
                     std::endl << "EPSG: " << _epsgcode << std::endl;
    }

    // Polar stereographic coordinate system
    struct PolarStereo : public ProjectionBase {
        // Constants related to projection system
        double lat0, lon0, lat_ts, akm1, e;
        bool isnorth;

        // Value constructor
        PolarStereo(int);

        inline void print() const;
        // Transfrom from LLH to Polar Stereo
        int forward(const cartesian_t&,cartesian_t&) const;
        // Transform from Polar Stereo to LLH
        int inverse(const cartesian_t&,cartesian_t&) const;
    };
    
    inline void PolarStereo::print() const {
        std::cout << "Projection: " << (isnorth ? "North" : "South") << " Polar Stereographic" <<
                     std::endl << "EPSG: " << _epsgcode << std::endl;
    }

    // Equal Area Projection System for SMAP
    struct CEA: public ProjectionBase {
        // Constants related to projection system
        double apa[3];
        double lat_ts, k0, e, one_es, qp;

        // Value constructor
        CEA();

        inline void print() const;
        // Transform from LLH to CEA
        int forward(const cartesian_t&,cartesian_t&) const;
        // Transform from CEA to LLH
        int inverse(const cartesian_t&,cartesian_t&) const;
    };

    inline void CEA::print() const {
        std::cout << "Projection: Cylindrical Equal Area" << std::endl << "EPSG: " << _epsgcode <<
                     std::endl;
    }

    //This is to create a projection system from the EPSG code
    ProjectionBase* createProj(int epsg);

    // This is to transform a point from one coordinate system to another
    int projTransform(ProjectionBase* in, ProjectionBase *out, const cartesian_t &inpts,
                      cartesian_t &outpts);
}}

#endif
