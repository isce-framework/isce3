
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

class isce::signal::Crossmul {
    public:
        // Constructor from product
        Crossmul() {};

        ~Crossmul() {};
        
        /*void crossmul(const isce::product::Product& referenceSLC,
                    const isce::product::Product& secondarySLC,
                    int numberOfRangeLooks,
                    int numberOfAzimuthLooks);
        */

        /* Do we need a constructro from raster?
        void Crossmul(isce::io::Raster& referenceSLC,
                      isce::io::Raster& secondarySLC,
                      int numberOfRangeLooks,
                      int numberOfAzimuthLooks,
                      isce::io::Raster& interferogram)
        */
        // Run crossmul 
        void crossmul(isce::io::Raster& referenceSLC, 
                      isce::io::Raster& secondarySLC,
                      int numberOfRangeLooks,
                      int numberOfAzimuthLooks,
                      double commonAzimuthBandwidth,
                      isce::io::Raster& interferogram);

        //estimateCommonAzimuthBandwidth();
        //estimateCommonRangeBandwidth();
       //void azimuthCommonBandFiltering();
       //void rangeCommonBandFiltering();
       //void crossmul();

    private:
        int nrows; 
        int ncols;
        int rowsLooks;
        int colsLooks;
        int nrows_ifgram;
        int ncols_ifgram;
        int nfft;

};

#endif
