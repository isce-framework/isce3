// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-
//

#ifndef ISCE_LIB_CROSSMUL_H
#define ISCE_LIB_CROSSMUL_H

# include <assert.h>

// pyre
#include <portinfo>
#include <pyre/journal.h>

#include <isce/io/Raster.h>
#include "Signal.h"

namespace isce {
    namespace signal {
        class Crossmul;
    }
}

/** \brief Intereferogram generation by cross-multiplication of reference and secondary SLCs.
 *
 *  The secondary SLC is expected to have been coregistered to the refernce SLC's grid.
 */
class isce::signal::Crossmul {
    public:
        // Constructor from product
        Crossmul() {};

        ~Crossmul() {};
        
        /*
        void Crossmul(const isce::product::Product& referenceSLC,
                    const isce::product::Product& secondarySLC,
                    int numberOfRangeLooks,
                    int numberOfAzimuthLooks,
                    const isce::product::Product& outputInterferogram);
        */


        /** \brief Run crossmul  
         *
         * @param[in] Raster object of refernce SLC
         * @param[in] Raster object of secondary SLC
         * @param[in] Raster object of output interferogram 
         * */
        void crossmul(isce::io::Raster& referenceSLC, 
                      isce::io::Raster& secondarySLC,
                      isce::io::Raster& interferogram);


       /** Set doppler polynomials dor reference and secondary SLCs*/
        inline void doppler(isce::core::Poly2d, 
                            isce::core::Poly2d);

        /** Set pulse repetition frequency (PRF) */
        inline void prf(double);

        /** Set azimuth common bandwidth */
        inline void commonAzimuthBandwidth(double);

        /** Set number of range looks */ 
        inline void rangeLooks(int);

        /** Set number of azimuth looks */
        inline void azimuthLooks(int);

        /** Set common azimuth band filtering flag */
        inline void doCommonAzimuthbandFiltering(bool);

        /** Set common range band filtering flag */
        inline void doCommonRangebandFiltering(bool);

    private:
        //Doppler polynomial for the refernce SLC
        isce::core::Poly2d _refDoppler;

        //Doppler polynomial for the secondary SLC
        isce::core::Poly2d _secDoppler;

        //pulse repetition frequency
        double _prf;

        //azimuth common bandwidth
        double _azimuthCommonBandwidth;

        // number of range looks
        int _rangeLooks;

        // number of azimuth looks
        int _azimuthLooks;

        // Flag for common azimuth band filtering
        bool _doCommonAzimuthbandFilter;

        // Flag for common range band filtering
        bool _doCommonRangebandFilter;

};

#endif
