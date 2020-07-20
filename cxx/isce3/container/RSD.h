#pragma once

#include <complex>
#include <vector>

#include <isce3/core/DateTime.h>
#include <isce3/core/LookSide.h>
#include <isce3/io/gdal/Raster.h>

namespace isce3 { namespace container {

/** Radar signal data */
class RSD {
public:
    /**
     * Constructor
     *
     * \param[in] signal_data           Radar signal (I/Q) data
     * \param[in] reference_epoch       Reference epoch (UTC)
     * \param[in] azimuth_time          Time of each azimuth line relative to reference epoch (s).
     *                                  Timepoints may be non-uniformly sampled but must be
     *                                  monotonically increasing.
     * \param[in] range_window_start    Time of first range sample relative to Tx pulse start (s)
     * \param[in] range_sampling_rate   Range sampling rate (Hz)
     * \param[in] look_side             Radar look side
     */
    RSD(const isce3::io::gdal::Raster & signal_data,
        const isce3::core::DateTime & reference_epoch,
        const std::vector<double> & azimuth_time,
        double range_window_start,
        double range_sampling_rate,
        isce3::core::LookSide look_side);

    /** Number of azimuth lines */
    int lines() const { return _signal_data.length(); }

    /** Number of range samples */
    int samples() const { return _signal_data.width(); }

    /** Reference epoch (UTC) */
    const isce3::core::DateTime & referenceEpoch() const { return _reference_epoch; }

    /** Time of first azimuth line relative to reference epoch (s) */
    double azimuthStartTime() const { return _azimuth_time[0]; }

    /** Time of center of data relative to reference epoch (s) */
    double azimuthMidTime() const { return 0.5 * azimuthStartTime() + 0.5 * azimuthEndTime(); }

    /** Time of last azimuth line relative to reference epoch (s) */
    double azimuthEndTime() const { return _azimuth_time[lines() - 1]; }

    /** DateTime of first azimuth line (UTC) */
    isce3::core::DateTime startDateTime() const;

    /** DateTime of center of data (UTC) */
    isce3::core::DateTime midDateTime() const;

    /** DateTime of last azimuth line (UTC) */
    isce3::core::DateTime endDateTime() const;

    /** Time of first range sample relative to Tx pulse start (s) */
    double rangeWindowStartTime() const { return _range_window_start; }

    /** Time of center of range sampling window relative to Tx pulse start (s) */
    double rangeWindowMidTime() const;

    /** Time of last range sample relative to Tx pulse start (s) */
    double rangeWindowEndTime() const;

    /** Range sampling rate (Hz) */
    double rangeSamplingRate() const { return _range_sampling_rate; }

    /** Radar look side */
    isce3::core::LookSide lookSide() const { return _look_side; }

    /** Read a single line of signal data */
    void readLine(std::complex<float> * dst, int line) const;

    /** Read one or more lines of signal data */
    void readLines(std::complex<float> * dst, int first_line, int num_lines) const;

    /** Read all signal data lines */
    void readAll(std::complex<float> * dst) const { return _signal_data.readAll(dst); }

    /** Get signal data Raster */
    const isce3::io::gdal::Raster & signalData() const { return _signal_data; }

    /** Get azimuth timepoints relative to reference epoch (s) */
    const std::vector<double> & azimuthTime() const { return _azimuth_time; }

private:
    isce3::io::gdal::Raster _signal_data;
    isce3::core::DateTime _reference_epoch;
    std::vector<double> _azimuth_time;
    double _range_window_start;
    double _range_sampling_rate;
    isce3::core::LookSide _look_side;
};

}}

#define ISCE_CONTAINER_RSD_ICC
#include "RSD.icc"
#undef ISCE_CONTAINER_RSD_ICC
