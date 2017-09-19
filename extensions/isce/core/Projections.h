//
// Author: Piyush Agram
// Copyright 2017
//
#ifndef __ISCE_CORE_PROJECTIONS_H__
#define __ISCE_CORE_PROJECTIONS_H__

#include <cmath>
#include <vector>
#include "isce/core/Constants.h"
#include "isce/core/Ellipsoid.h"

namespace isce { namespace core {

    //Base class for individual projections
    struct ProjectionBase {
        //Ellipsoid to be used for all transformations
        Ellipsoid ellipse;

        //Type of projection system
        //This can be used to check if projection systems are equal
        //Private member and should not be modified after initialization
        int epsgcode;

        //Print function for debugging
        virtual void print()= 0 ;
    
        //Function for transforming from LLH
        //This is similar to fwd or fwd3d in PROJ.4
        virtual int forward( std::vector<double>&,
                std::vector<double>&) = 0 ;

        //Function for transforming to LLH
        //This is similar to inv or inv3d in PROJ.4
        virtual int inverse( std::vector<double>&,
                std::vector<double>&) = 0 ;
      
        //ructor with Ellipsoid as input
        ProjectionBase(Ellipsoid &elp, int code):ellipse(elp), epsgcode(code){};

    };

    //Standard WGS84 Lat/Lon 
    struct LatLon : public ProjectionBase {
        virtual void print() ;

        //This will be a pass through for Lat/Lon
        virtual int forward( std::vector<double>& in, std::vector<double>& out) ;

        //This will also be a pass through for Lat/Lon
        virtual int inverse( std::vector<double>& in, std::vector<double>& out) ;

        //ructor
        LatLon(Ellipsoid &elp):ProjectionBase(elp,4326){};

    };

    //Standard WGS84 ECEF coordinates
    struct Geocent : public ProjectionBase {
        void print() ;

        //This is same as LLH2XYZ
        virtual int forward( std::vector<double>&, std::vector<double>&) ;

        //This is same as XYZ2LLH
        virtual int inverse( std::vector<double>&, std::vector<double>&) ;

        //ructor
        Geocent(Ellipsoid &elp):ProjectionBase(elp, 4978){};

    };

    //UTM coordinates
    struct UTM : public ProjectionBase {
        
        //ants related to the projection system
        bool isnorth;
        double lon0;
        int zone;
        
        //Parameters from Proj.4
        double Qn;
        double Zb; 
        double cgb[6];
        double cbg[6]; 
        double utg[6];
        double gtu[6];

        virtual void print() ;

        //Transform from LLH to UTM
        virtual int forward( std::vector<double>&, std::vector<double>&) ;

        //Transform from UTM to LLH
        virtual int inverse( std::vector<double>&, std::vector<double>&) ;

        //Private methods. Not part of public interface.
        void setup();

        //ructor
        UTM(Ellipsoid &elp, int code):ProjectionBase(ellipse,code){ setup(); };

    };

    //Polar stereographic coordinate system
    struct PolarStereo : public ProjectionBase {
        //ants related to projection system
        bool isnorth;
        double lat0;
        double lon0;
        double lat_ts;
        double akm1;
        double e;

        virtual void print() ;

        //Transfrom from LLH to Polar Stereo
        virtual int forward( std::vector<double>&, std::vector<double> &) ;

        //Transform from Polar Stereo to LLH
        virtual int inverse( std::vector<double>&, std::vector<double> &) ;

        //Private methods. Not part of public interface.
        void setup();

        //ructor
        PolarStereo(Ellipsoid &elp, int code): ProjectionBase(elp,code){ setup(); };

    };


    /*******************General functions - user interface***********/
    //This is the factory for generating a projection transformer
    ProjectionBase* createProj(int epsgcode);

    //This is to transform a point from one coordinate system
    //to another
    int projTransform(ProjectionBase* in, ProjectionBase *out,
                   std::vector<double> &inpts,
                  std::vector<double> &outpts);

}}

#endif
