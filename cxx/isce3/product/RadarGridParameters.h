//-*- C++ -*-
//-*- coding: utf-8 -*-

#pragma once

#include <cmath>
#include "forward.h"

// isce3::core
#include <isce3/core/Metadata.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/TimeDelta.h>
#include <isce3/except/Error.h>

class isce3::product::RadarGridParameters {

    public:
        /** Default constructor */
        inline RadarGridParameters();

        /**
         * Constructor with a product
         * @param[in] product Input RadarGridProduct
         * @param[in] frequency Frequency designation
         */
        RadarGridParameters(const isce3::product::RadarGridProduct & product,
                            char frequency = 'A');

        /**
         * Constructor with a swath
         * @param[in] swath Input swath
         * @param[in] lookSide Indicate left (+1) or right (-1)
         */
        RadarGridParameters(const isce3::product::Swath & swath,
                            isce3::core::LookSide lookSide);

        /** Constructor from an isce3::core::Metadata object. */
        inline RadarGridParameters(const isce3::core::Metadata & meta,
                                   const isce3::core::DateTime & refEpoch);

        /** Constructor from individual components and values. */
        inline RadarGridParameters(double sensingStart,
                                   double wavelength,
                                   double prf,
                                   double startingRange,
                                   double rangePixelSpacing,
                                   isce3::core::LookSide lookSide,
                                   size_t length,
                                   size_t width,
                                   isce3::core::DateTime refEpoch);

        /** Copy constructor */
        inline RadarGridParameters(const RadarGridParameters & rgparam);

        /** Assignment operator */
        inline RadarGridParameters&
        operator=(const RadarGridParameters& rgparam);

        /** Get the look direction */
        inline isce3::core::LookSide lookSide() const { return _lookSide; }

        /** Set look direction */
        inline void lookSide(isce3::core::LookSide side) { _lookSide = side; }

        /** Set look direction from a string */
        inline void lookSide(const std::string &);

        /** Get reference epoch DateTime*/
        inline const isce3::core::DateTime & refEpoch() const { return _refEpoch; }

        /** Set reference epoch DateTime
         *
         * Other dependent parameters like sensingStart are not modified. Use with caution.*/
        inline void refEpoch(const isce3::core::DateTime &epoch) { _refEpoch = epoch; }

        /** Get sensing start time in seconds since reference epoch */
        inline double sensingStart() const { return _sensingStart; }

        /** Set sensing start time in seconds since reference epoch */
        inline void sensingStart(const double & t){ _sensingStart = t; }

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

        /** Set starting slant range in meters*/
        inline void startingRange(const double & r){ _startingRange = r; }

        /** Get slant range pixel spacing in meters*/
        inline double rangePixelSpacing() const { return _rangePixelSpacing; }

        /** Set slant range pixel spacing in meters */
        inline void rangePixelSpacing(const double & t) { _rangePixelSpacing = t; }

        /** Get radar grid length */
        inline size_t length() const { return _rlength; }

        /** Set radar grid length */
        inline void length(const size_t & t) { _rlength = t; }

        /** Get radar grid width */
        inline size_t width() const { return _rwidth; }

        /** Set radar grid width */
        inline void width(const size_t & t) { _rwidth = t; }

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
        /** Get azimuth fractional index (line) at a given sensing time */
        inline double azimuthIndex(double az_time) const {
            return (az_time  -  _sensingStart) * _prf;
        }

        /** Get a sensing DateTime for a given line (zero-index row) */
        inline isce3::core::DateTime sensingDateTime(double line) const {
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

        /** Get slant range fractional index at a given slant range distance */
        inline double slantRangeIndex(double slant_range) const {
            return (slant_range  -  _startingRange) / _rangePixelSpacing;
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

        /** Return a new resized radar grid with the same start and stop points.
         * @param[in] ysize  new number of samples along the azimuth direction
         * @param[in] xsize  new number of samples along the slant range direction
         */
        inline RadarGridParameters resizeKeepStartStop(size_t ysize,
                                                       size_t xsize) const
        {
            //Check for number of samples requested along the azimuth and slant range
            if ((ysize <= 1) || (xsize <= 1))
            {
                std::string errstr = "Number of samples must be greater than 1. " +
                                    std::to_string(ysize) + " azimuth and " +
                                    std::to_string(xsize) + " range samples requested.";
                throw isce3::except::OutOfRange(ISCE_SRCINFO(), errstr);
            }

            // Compute the new PRF and range pixel spacing to ensure that
            // both start and stop points are kept
            double prf = this->prf();
            double dr = rangePixelSpacing();

            if (length() > 1) prf = (ysize - 1.0) * prf / (length() - 1.0) ;
            dr = (width() - 1.0) * dr / (xsize - 1.0) ;

            return RadarGridParameters(sensingStart(), wavelength(), prf,
                                       startingRange(), dr, lookSide(),
                                       ysize, xsize, refEpoch());
        }

        /** Add margins to start, stop or both sides to the radar grid
         * @param[in] ymargin  the number of margin samples along the azimuth
         * @param[in] xmargin  the number of margin samples along the slant range
         * @param[in] side  the side where the margin samples will be added ('start', 'stop', 'both')
         * NOTE: When only add the margin to the azimuth, please pass 'xmargin' as 0.
         *       Similarly, when only add the margin to the slant range, please pass 'ymargin' as 0.
         */
        inline RadarGridParameters addMargin(size_t ymargin,
                                             size_t xmargin,
                                             std::string side = "both") const
        {
            double t0, r0;
            size_t ysize, xsize;

            if (side == "start") {
                ysize = length() + ymargin;
                xsize = width() + xmargin;
                t0 = sensingStart() - ymargin / prf();
                r0 = startingRange() - xmargin * rangePixelSpacing();
            } else if (side == "stop") {
                ysize = length() + ymargin;
                xsize = width() + xmargin;
                t0 = sensingStart();
                r0 = startingRange();
            } else if (side == "both") {
                t0 = sensingStart() - ymargin / prf();
                r0 = startingRange() - xmargin * rangePixelSpacing();
                ysize = length() + 2 * ymargin;
                xsize = width() + 2 * xmargin;
            } else {
                std::string errstr = "No recognized side '" + side +
                                     "', please choose the side from 'start', 'stop', and 'both'";
                throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errstr);
            }

            return RadarGridParameters(t0, wavelength(), prf(),
                                       r0, rangePixelSpacing(), lookSide(),
                                       ysize, xsize, refEpoch());
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
                throw isce3::except::OutOfRange(ISCE_SRCINFO(), errstr);
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

        /** Upsample */
        inline RadarGridParameters
        upsample(size_t az_upsampling_factor, size_t rg_upsampling_factor) const
        {
            // Check for number of points on edge
            if ((az_upsampling_factor  == 0) || (rg_upsampling_factor  == 0)) {
                std::string errstr = "Upsampling factor must be positive. " +
                                     std::to_string(az_upsampling_factor ) + "Az x" +
                                     std::to_string(rg_upsampling_factor ) +
                                     "Rg upsampling requested.";
                throw isce3::except::OutOfRange(ISCE_SRCINFO(), errstr);
            }

            // important: differently from multilook(), upsample does not
            // update _sensingStart or _startingRange
            return RadarGridParameters(
                        sensingTime(0.0), wavelength(),
                        prf() * az_upsampling_factor, slantRange(0.0),
                        rangePixelSpacing() / (1.0 * rg_upsampling_factor),
                        lookSide(), length() * az_upsampling_factor,
                        width() * rg_upsampling_factor, refEpoch());
        }

        /*
         * Check if given az and slant range are within radargrid
         */
        bool contains(const double az_time, const double srange) const;

    // Protected data members can be accessed by derived classes
    protected:
        /** Left or right looking geometry indicator */
        isce3::core::LookSide _lookSide;

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
        isce3::core::DateTime _refEpoch;

        /** Validate parameters of data structure */
        inline void validate() const;
};

isce3::product::RadarGridParameters::RadarGridParameters()
    : _lookSide(isce3::core::LookSide::Left), _sensingStart {0},
      _wavelength {0}, _prf {0}, _startingRange {0},
      _rangePixelSpacing {0}, _rlength {0}, _rwidth {0}, _refEpoch {1} {}

// Constructor from an isce3::core::Metadata object.
/** @param[in] meta isce3::core::Metadata object
  * @param[in] refEpoch Reference epoch date */
isce3::product::RadarGridParameters::
RadarGridParameters(const isce3::core::Metadata & meta,
                    const isce3::core::DateTime & refEpoch) :
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
isce3::product::RadarGridParameters::
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
isce3::product::RadarGridParameters &
isce3::product::RadarGridParameters::
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
isce3::product::RadarGridParameters::
RadarGridParameters(double sensingStart,
                    double wavelength,
                    double prf,
                    double startingRange,
                    double rangePixelSpacing,
                    isce3::core::LookSide lookSide,
                    size_t length,
                    size_t width,
                    isce3::core::DateTime refEpoch) :
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
isce3::product::RadarGridParameters::
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
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errstr);
    }
}


/** @param[in] look String representation of look side */
void
isce3::product::RadarGridParameters::
lookSide(const std::string & inputLook) {
    _lookSide = isce3::core::parseLookSide(inputLook);
}
