//-*- C++ -*-
//-*- coding: utf-8 -*-

#pragma once

// isce::core
#include <isce/core/Metadata.h>
#include <isce/core/DateTime.h>
#include <isce/core/LookSide.h>
#include <isce/core/TimeDelta.h>

// isce::product
#include <isce/product/Product.h>

namespace isce {
    namespace product {
        class RadarGridParameters; 
    }
}

class isce::product::RadarGridParameters {

    public:
        /** Default constructor */
        inline RadarGridParameters() {}

        /** Constructor with a product */
        inline RadarGridParameters(const isce::product::Product & product,
                                   char frequency = 'A');

        /** Constructor with a swath. */
        inline RadarGridParameters(const isce::product::Swath & swath,
                                   isce::core::LookSide lookSide);

        /** Constructor from an isce::core::Metadata object. */
        inline RadarGridParameters(const isce::core::Metadata & meta,
                                   const isce::core::DateTime & refEpoch);

        /** Constructor from individual components and values. */
        inline RadarGridParameters(double sensingStart,
                                   double wavelength,
                                   double prf,
                                   double startingRange,
                                   double rangePixelSpacing,
                                   isce::core::LookSide lookSide,
                                   size_t length,
                                   size_t width,
                                   isce::core::DateTime refEpoch);

        /** Copy constructor */
        inline RadarGridParameters(const RadarGridParameters & rgparam);

        /** Assignment operator */
        inline RadarGridParameters & operator=(const RadarGridParameters & rgparam);

        /** Get the look direction */
        inline isce::core::LookSide lookSide() const { return _lookSide; }

        /** Set look direction */
        inline void lookSide(isce::core::LookSide side) { _lookSide = side; }

        /** Set look direction from a string */
        inline void lookSide(const std::string &);

        /** Get reference epoch DateTime*/
        inline const isce::core::DateTime & refEpoch() const { return _refEpoch; }

        /** Get sensing start time in seconds since reference epoch */
        inline double sensingStart() const { return _sensingStart; }

        /** Get radar wavelength in meters*/
        inline double wavelength() const { return _wavelength; }

        /** Set radar wavelength in meters*/
        inline void wavelength(const double & t) { _wavelength = t; }

        /** Get pulse repetition frequency in Hz - inverse of azimuth time interval*/
        inline double prf() const { return _prf; }
        
        /** Set pulse repetition frequency in Hz - inverse of azimuth time interval*/
        inline void prf(const double & t){ _prf = t; }

        /** Get azimuth time interval in seconds*/
        inline double azimuthTimeInterval() const { return 1.0/_prf; };

        /** Get starting slant range in meters*/
        inline double startingRange() const { return _startingRange; }

        /** Get slant range pixel spacing in meters*/
        inline double rangePixelSpacing() const { return _rangePixelSpacing; }

        /** Set slant range pixel spacing in meters */
        inline void rangePixelSpacing(const double & t) { _rangePixelSpacing = t; }

        /** Get radar grid length */
        inline size_t length() const { return _rlength; }
        
        /** Set radar grid length */
        inline void length(const double & t) { _rlength = t; }

        /** Get radar grid width */
        inline size_t width() const { return _rwidth; }
        
        /** Set radar grid length */
        inline void width(const double & t) { _rwidth = t; }

        /** Get total number of radar grid elements */
        inline size_t size() const { return _rlength * _rwidth; }

        /** Get sensing stop time in seconds since reference epoch*/
        inline double sensingStop() const { return sensingTime(_rlength - 1.0); }

        /** Get sensing mid time in seconds */
        inline double sensingMid() const {
            return 0.5 * (sensingStart() + sensingStop());
        }

        /** Get sensing time for a given line (zero-index row) */
        inline double sensingTime(double line) const {
            return _sensingStart + line / _prf;
        }

        /** Get a sensing DateTime for a given line (zero-index row) */
        inline isce::core::DateTime sensingDateTime(double line) const {
            return _refEpoch + sensingTime(line);
        }

        /** Get ending slant range */
        inline double endingRange() const {
            return slantRange(_rwidth - 1.0);
        }

        /** Get middle slant range */
        inline double midRange() const {
            return 0.5 * (startingRange() + endingRange());
        }

        /** Get slant range for a given sample (zero-index column) */
        inline double slantRange(double sample) const {
            return _startingRange + sample * _rangePixelSpacing;
        }
        
        /** Crop/ Expand while keeping the spacing the same with top left offset and size */
        inline RadarGridParameters offsetAndResize(double yoff, double xoff, size_t ysize, size_t xsize) const
        {
            return RadarGridParameters( sensingTime(yoff),
                                        wavelength(),
                                        prf(),
                                        slantRange(xoff),
                                        rangePixelSpacing(),
                                        lookSide(),
                                        ysize,
                                        xsize,
                                        refEpoch());
        }


        /** Multilook */
        inline RadarGridParameters multilook(size_t azlooks, size_t rglooks) const
        {
                //Check for number of points on edge
            if ((azlooks == 0) || (rglooks == 0))
            {
                std::string errstr = "Number of looks must be positive. " +
                                    std::to_string(azlooks) + "Az x" +
                                    std::to_string(rglooks) + "Rg looks requested.";
                throw isce::except::OutOfRange(ISCE_SRCINFO(), errstr); 
            }

            //Currently implements the multilooking operation used in ISCE2 
            return RadarGridParameters( sensingTime(0.5 * (azlooks-1)),
                                        wavelength(),
                                        prf() / (1.0 * azlooks),
                                        slantRange(0.5 * (rglooks-1)),
                                        rangePixelSpacing() * rglooks,
                                        lookSide(),
                                        length()/azlooks,
                                        width()/rglooks,
                                        refEpoch());
        }

    // Protected data members can be accessed by derived classes
    protected:
        /** Left or right looking geometry indicator */
        isce::core::LookSide _lookSide;

        /** Sensing start time */
        double _sensingStart;

        /** Imaging wavelength */
        double _wavelength;

        /** PRF */
        double _prf;

        /** Slant range to center of first pixel */
        double _startingRange;

        /** Slant range pixel spacing */
        double _rangePixelSpacing;

        /** Number of lines in the image */
        size_t _rlength;

        /** Number of samples in the image */
        size_t _rwidth;

        /** Reference epoch for time tags */
        isce::core::DateTime _refEpoch;

        /** Validate parameters of data structure */
        inline void validate() const;
};

// Constructor with a swath.
/** @param[in] swath Input swath
  * @param[in] lookSide Indicate left (+1) or right (-1)*/
isce::product::RadarGridParameters::
RadarGridParameters(const isce::product::Swath & swath,
                    isce::core::LookSide lookSide) :
    _lookSide(lookSide),
    _sensingStart(swath.zeroDopplerTime()[0]),
    _wavelength(swath.processedWavelength()),
    _prf(swath.nominalAcquisitionPRF()),
    _startingRange(swath.slantRange()[0]),
    _rangePixelSpacing(swath.rangePixelSpacing()),
    _rlength(swath.lines()),
    _rwidth(swath.samples()),
    _refEpoch(swath.refEpoch()){ validate(); }

// Constructor with a product
/** @param[in] product Input Product
  * @param[in] frequency Frequency designation */
isce::product::RadarGridParameters::
RadarGridParameters(const isce::product::Product & product,
                    char frequency) :
    RadarGridParameters(product.swath(frequency), product.lookSide()){ validate(); }

// Constructor from an isce::core::Metadata object.
/** @param[in] meta isce::core::Metadata object
  * @param[in] refEpoch Reference epoch date */
isce::product::RadarGridParameters::
RadarGridParameters(const isce::core::Metadata & meta,
                    const isce::core::DateTime & refEpoch) :
    _lookSide(meta.lookSide),
    _sensingStart((meta.sensingStart - refEpoch).getTotalSeconds()),
    _wavelength(meta.radarWavelength),
    _prf(meta.prf),
    _startingRange(meta.rangeFirstSample),
    _rangePixelSpacing(meta.slantRangePixelSpacing),
    _rlength(meta.length),
    _rwidth(meta.width),
    _refEpoch(refEpoch) { validate(); }

// Copy constructors
/** @param[in] rgparam RadarGridParameters object */
isce::product::RadarGridParameters::
RadarGridParameters(const RadarGridParameters & rgparams) :
    _lookSide(rgparams.lookSide()),
    _sensingStart(rgparams.sensingStart()),
    _wavelength(rgparams.wavelength()),
    _prf(rgparams.prf()),
    _startingRange(rgparams.startingRange()),
    _rangePixelSpacing(rgparams.rangePixelSpacing()),
    _rlength(rgparams.length()),
    _rwidth(rgparams.width()),
    _refEpoch(rgparams.refEpoch()) { validate(); }

// Assignment operator
/** @param[in] rgparam RadarGridParameters object */
isce::product::RadarGridParameters &
isce::product::RadarGridParameters::
operator=(const RadarGridParameters & rgparams) {
    _sensingStart = rgparams.sensingStart();
    _wavelength = rgparams.wavelength();
    _prf = rgparams.prf();
    _startingRange = rgparams.startingRange();
    _rangePixelSpacing = rgparams.rangePixelSpacing();
    _lookSide = rgparams.lookSide(),
    _rlength = rgparams.length();
    _rwidth = rgparams.width();
    _refEpoch = rgparams.refEpoch();
    validate();
    return *this;
}

// Constructor from individual components and values
isce::product::RadarGridParameters::
RadarGridParameters(double sensingStart,
                    double wavelength,
                    double prf,
                    double startingRange,
                    double rangePixelSpacing,
                    isce::core::LookSide lookSide,
                    size_t length,
                    size_t width,
                    isce::core::DateTime refEpoch) :
    _lookSide(lookSide),
    _sensingStart(sensingStart),
    _wavelength(wavelength),
    _prf(prf),
    _startingRange(startingRange),
    _rangePixelSpacing(rangePixelSpacing),
    _rlength(length),
    _rwidth(width),
    _refEpoch(refEpoch) { validate(); }

// Validation of radar grid parameters
void
isce::product::RadarGridParameters::
validate() const
{
    std::string errstr = "";

    if (wavelength() <= 0.)
    {
        errstr += "Radar wavelength must be positive. \n";
    }

    if (prf() <= 0.)
    {
        errstr += "Pulse Repetition Frequency must be positive. \n";
    }

    if (startingRange() <= 0.)
    {
        errstr += "Starting Range must be positive. \n";
    }

    if (rangePixelSpacing() <= 0. )
    {
        errstr += "Slant range pixel spacing must be positive. \n";
    }

    if (length() == 0)
    {
        errstr += "Radar Grid should have length of at least 1. \n";
    }

    if (width() == 0)
    {
        errstr += "Radar Grid should have width of at least 1. \n";
    }

    if (! errstr.empty())
    {
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), errstr);
    }
}


/** @param[in] look String representation of look side */
void
isce::product::RadarGridParameters::
lookSide(const std::string & inputLook) {
    _lookSide = isce::core::parseLookSide(inputLook);
}
