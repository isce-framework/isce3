//-*- C++ -*-
//-*- coding: utf-8 -*-

#pragma once

// std
#include <valarray>

// isce3::core
#include <isce3/core/DateTime.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Constants.h>

// isce3::io
#include <isce3/io/Raster.h>

// isce3::product
#include <isce3/product/GeoGridParameters.h>

// Declaration
namespace isce3 {
    namespace product {


/**
 * A class for representing Grid metadata originally based on
 NISAR L2 products.
 */
class Grid {

    public:
        // Constructors
        inline Grid() {};

        /** Get acquired range bandwidth in Hz */
        inline double rangeBandwidth() const { return _rangeBandwidth; }
        /** Set acquired range bandwidth in Hz */
        inline void rangeBandwidth(double b) { _rangeBandwidth = b; }

        /** Get acquired azimuth bandwidth in Hz */
        inline double azimuthBandwidth() const { return _azimuthBandwidth; }
        /** Set acquired azimuth bandwidth in Hz */
        inline void azimuthBandwidth(double b) { _azimuthBandwidth = b; }

        /** Get processed center frequency in Hz */
        inline double centerFrequency() const { return _centerFrequency; }
        /** Set processed center frequency in Hz */
        inline void centerFrequency(double f) { _centerFrequency = f; }

        /** Get processed wavelength in meters */
        inline double wavelength() const {
            return isce3::core::speed_of_light / _centerFrequency;
        }

        /** Get scene center ground range spacing in meters */
        inline double slantRangeSpacing() const {
            return _slantRangeSpacing;
        }
        /** Set scene center ground range spacing in meters */
        inline void slantRangeSpacing(double s) {
            _slantRangeSpacing = s;
        }

        /** Get geogrid */
        inline isce3::product::GeoGridParameters geogrid() { 
            return _geogrid; 
        }

        /** Set geogrid */
        inline void geogrid(isce3::product::GeoGridParameters geogrid) { 
            _geogrid = geogrid;
        }

        /** Get time spacing of raster grid in seconds */
        inline double zeroDopplerTimeSpacing() const { return _zeroDopplerTimeSpacing; }
        /** Set time spacing of raster grid in seconds */
        inline void zeroDopplerTimeSpacing(double dt) { _zeroDopplerTimeSpacing = dt; }

        /* Geogrid parameters */

        /** Get the X-coordinate start */
        inline double startX() const { return _geogrid.startX(); }
        /** Set the X-coordinate start */
        inline void startX(double val) { _geogrid.startX(val); }

        /** Get the y-coordinate start */
        inline double startY() const { return _geogrid.startY(); }
        /** Set the y-coordinate start */
        inline void startY(double val) { _geogrid.startY(val);}

        /** Get the X-coordinate spacing */
        inline double spacingX() const { return _geogrid.spacingX(); }
        /** Set the X-coordinate spacing */
        inline void spacingX(double val) { _geogrid.spacingX(val); }

        /** Get the y-coordinate spacing */
        inline double spacingY() const { return _geogrid.spacingY(); }
        /** Set the y-coordinate spacing */
        inline void spacingY(double val) { _geogrid.spacingY(val);}

        /** Get number of pixels in east-west/x direction for geocoded grid */
        inline size_t width() const { return _geogrid.width(); }
        /** Set number of pixels in north-south/y direction for geocoded grid */
        inline void width(int w) { _geogrid.width(w); }

        /** Get number of pixels in north-south/y direction for geocoded grid */
        inline size_t length() const { return _geogrid.length(); }
        //** Set number of pixels in east-west/x direction for geocoded grid */
        inline void length(int l) { _geogrid.length(l); }

        /** Get epsg code for geocoded grid */
        inline size_t epsg() const { return _geogrid.epsg(); }
        //** Set epsg code for geocoded grid */
        inline void epsg(int l) { _geogrid.epsg(l); }


    private:

        // Other metadata
        isce3::product::GeoGridParameters _geogrid;
        double _rangeBandwidth = std::numeric_limits<double>::quiet_NaN();
        double _azimuthBandwidth = std::numeric_limits<double>::quiet_NaN();
        double _slantRangeSpacing = std::numeric_limits<double>::quiet_NaN();
        double _zeroDopplerTimeSpacing = std::numeric_limits<double>::quiet_NaN();
        double _centerFrequency = std::numeric_limits<double>::quiet_NaN();

};


    }
}