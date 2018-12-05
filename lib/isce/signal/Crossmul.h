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
#include <isce/core/LUT1d.h>
#include "Signal.h"
#include "Filter.h"


namespace isce {
    namespace signal {
        class Crossmul;
    }
}

/** \brief Intereferogram generation by cross-multiplication of reference and secondary SLCs.
 *
 *  The secondary SLC must be on the same image grid as the reference SLC, 
 */
class isce::signal::Crossmul {
    public:
        // Constructor from product
        Crossmul() {};

        ~Crossmul() {};
        
        /*
        void Crossmul(const isce::product::Product& referenceSLC,
                    const isce::product::Product& secondarySLC,
                    const isce::product::Product& outputInterferogram);
        */


        /** \brief Run crossmul */
        void crossmul(isce::io::Raster& referenceSLC, 
                      isce::io::Raster& secondarySLC,
                      isce::io::Raster& interferogram);


       /** Set doppler LUTs for reference and secondary SLCs*/
        inline void doppler(isce::core::LUT1d<double>, 
                            isce::core::LUT1d<double>);

        /** Set pulse repetition frequency (PRF) */
        inline void prf(double);

        /** Set azimuth common bandwidth */
        inline void commonAzimuthBandwidth(double);

        /** Set beta parameter for the azimuth common band filter */
        inline void beta(double);


        /** Set number of range looks */ 
        inline void rangeLooks(int);

        /** Set number of azimuth looks */
        inline void azimuthLooks(int);

        /** Set common azimuth band filtering flag */
        inline void doCommonAzimuthbandFiltering(bool);

        /** Set common range band filtering flag */
        inline void doCommonRangebandFiltering(bool);

    private:
        //Doppler LUT for the refernce SLC
        isce::core::LUT1d<double> _refDoppler;

        //Doppler LUT for the secondary SLC
        isce::core::LUT1d<double> _secDoppler;

        //pulse repetition frequency
        double _prf;

        //azimuth common bandwidth
        double _commonAzimuthBandwidth;

        // beta parameter for constructing common azimuth band filter
        double _beta;

        // number of range looks
        int _rangeLooks;

        // number of azimuth looks
        int _azimuthLooks;

        // Flag for common azimuth band filtering
        bool _doCommonAzimuthbandFilter;

        // Flag for common range band filtering
        bool _doCommonRangebandFilter;

        // number of lines per block
        size_t blockRows = 1000;

        // upsampling factor
        size_t oversample = 2;
};

// Get inline implementations for Crossmul
#define ISCE_SIGNAL_CROSSMUL_ICC
#include "Crossmul.icc"
#undef ISCE_SIGNAL_CROSSMUL_ICC

#endif
