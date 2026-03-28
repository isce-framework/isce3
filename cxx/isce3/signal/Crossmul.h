#pragma once

#include "forward.h"

#include <complex>
#include <isce3/core/Constants.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/forward.h>
#include <isce3/io/forward.h>

/** \brief Intereferogram generation by cross-multiplication of reference and secondary SLCs.
 *
 *  The secondary SLC must be on the same image grid as the reference SLC,
 */
class isce3::signal::Crossmul {
    public:
        // Constructor from product
        Crossmul() {};

        ~Crossmul() {};

        /**
         * Crossmultiply 2 SLCs
         *
         * \param[in]  refSlcRaster input raster of reference SLC
         * \param[in]  secSlcRaster input raster of secondary SLC
         * \param[out] ifgRaster    output interferogram raster
         * \param[out] coherenceRaster  output coherence raster
         * \param[in]  rngOffsetRaster  optional pointer to range offset raster, in pixels
         * \param[in]  aziOffsetRaster  optional pointer to azimuth offset raster, in pixels
         *
         */
        void crossmul(isce3::io::Raster& refSlcRaster,
                    isce3::io::Raster& secSlcRaster,
                    isce3::io::Raster& ifgRaster,
                    isce3::io::Raster& coherence,
                    isce3::io::Raster* rngOffsetRaster = nullptr,
                    isce3::io::Raster* aziOffsetRaster = nullptr);

        /** Set doppler LUTs for reference and secondary SLCs*/
        inline void doppler(isce3::core::LUT2d<double>,
                            isce3::core::LUT2d<double>);

        /** Set dopplers LUT for reference SLC */
        inline void refDoppler(isce3::core::LUT2d<double> refDopp) { _refDoppler = refDopp; }

        /** Get doppler LUT for reference SLC */
        inline const isce3::core::LUT2d<double> & refDoppler() const { return _refDoppler; }

        /** Set dopplers LUT for secondary SLC */
        inline void secDoppler(isce3::core::LUT2d<double> secDopp) { _secDoppler = secDopp; }

        /** Get doppler LUT for secondary SLC */
        inline const isce3::core::LUT2d<double> & secDoppler() const { return _secDoppler; }

        /** Set reference and seconary starting range shift */
        inline void startingRangeShift(double rng_shift) { _offsetStartingRangeShift = rng_shift; }

        /** Get reference and secondary starting range shift */
        inline double startingRangeShift() const { return _offsetStartingRangeShift; }

        /** Set range pixel spacing, in meters */
        inline void rangePixelSpacing(double rgPxlSpacing) { _rangePixelSpacing = rgPxlSpacing; }

        /** Get range pixel spacing, in meters */
        inline double rangePixelSpacing() const { return _rangePixelSpacing; }

        /** Set start range for reference and secondary, in meters */
        inline void refStartRange(double startRng) { _refStartRange = startRng; }
        inline void secStartRange(double startRng) { _secStartRange = startRng; }

        /** Get start range for reference and secondary , in meters */
        inline double refStartRange() const { return _refStartRange; }
        inline double secStartRange() const { return _secStartRange; }

        /** Set start azimuth time for reference and secondary, in seconds */
        inline void refStartAzimuthTime(double startAziTime) { _refStartAzimuthTime = startAziTime; }
        inline void secStartAzimuthTime(double startAziTime) { _secStartAzimuthTime = startAziTime; }

        /** Get start azimuth time for reference and secondary, in seconds */
        inline double refStartAzimuthTime() const { return _refStartAzimuthTime; }
        inline double secStartAzimuthTime() const { return _secStartAzimuthTime; }

        /** Set Wavelength, in meters*/
        inline void wavelength(double wvl) { _wavelength = wvl; }

        /** Get Wavelength*/
        inline double wavelength() const { return _wavelength; }

        /** Set number of range looks */
        inline void rangeLooks(int);

        /** Get number of range looks */
        inline int rangeLooks() const { return _rangeLooks; }

        /** Set number of azimuth looks */
        inline void azimuthLooks(int);

        /** Get number of azimuth looks */
        inline int azimuthLooks() const { return _azimuthLooks; }

        /** Set common azimuth band filtering flag */
        inline void doCommonAzimuthBandFilter(bool doAzBandFilter) {
            _doCommonAzimuthBandFilter = doAzBandFilter; }

        /** Get common azimuth band filtering flag */
        inline bool doCommonAzimuthBandFilter() const {
            return _doCommonAzimuthBandFilter; }

        /** Set azimuth bandwidth, in Hz*/
        inline void azimuthBandwidth(double azBandwidth) {
            _azimuthBandwidth = azBandwidth; }

        /** Get azimuth bandwidth, in Hz */
        inline double azimuthBandwidth() const {
            return _azimuthBandwidth; }

        /** Get processed azimuth bandwidth, in Hz.
         * This is the average azimuth bandwidth after
         * common-band filtering from among all blocks.
         */
        inline double processedAzimuthBandwidth() const {
            return _processedAzimuthBandwidth; }

        /** Set window parameter for the common band filter
         The meaning of this parameter depends on the `window_type`.
         For a raised-cosine window, it is the pedestal height of the window.
         For a Kaiser window, it is the beta parameter.*/
        inline void windowParameter(double windowParameter) { _windowParameter = windowParameter; }

        /** Get window parameter for the common band filter */
        inline double windowParameter() const { return _windowParameter; }

        /** Get window type for the common band filter */
        inline std::string windowType() const { return _windowType; }

        /** Set the window type */
        inline void windowType(std::string windowType) { _windowType = windowType; }

        /** Get sensor type*/
        inline std::string sensorType() const { return _sensorType; }

        /** Set the sensor type */
        inline void sensorType(std::string sensorType) { _sensorType = sensorType;}

        /** Set common range band filtering flag */
        inline void doCommonRangeBandFilter(bool doRgBandFilter) {
            _doCommonRangeBandFilter = doRgBandFilter; }

        /** Get common range band filtering flag */
        inline bool doCommonRangeBandFilter() const {
            return _doCommonRangeBandFilter; }

        /** Set flatten flag */
        inline void doFlatten(bool doFlatten) {
            _doFlatten = doFlatten; }

        /** Get flatten flag */
        inline bool doFlatten() const {
            return _doFlatten; }

        /** Set pulse repetition frequency (PRF), in Hz */
        inline void prf(double prf) { _prf = prf; }

        /** Get pulse repetition frequency (PRF), in Hz */
        inline double prf() const { return _prf; }

        /** Set the range bandwidth in Hz*/
        inline void rangeBandwidth(double rngBandwidth) { _rangeBandwidth = rngBandwidth; }

        /** Get the range bandwidth, in Hz */
        inline double rangeBandwidth() const {return _rangeBandwidth; }

        /** Get processed range bandwidth after common band filter
         * This is the average range bandwidth after
         * common-band filtering from among all blocks.
        */
        inline double processedRangeBandwidth() const {
            return _processedRangeBandwidth; }

        /** Set oversample factor */
        inline void oversampleFactor(size_t oversamp) { _oversampleFactor = oversamp; }

        /** Get oversample factor */
        inline size_t oversampleFactor() const { return _oversampleFactor; }

        /** Set linesPerBlock */
        inline void linesPerBlock(size_t linesPerBlock) { _linesPerBlock = linesPerBlock; }

        /** Get linesPerBlock */
        inline size_t linesPerBlock() const { return _linesPerBlock; }

        /** Get boolean multilook flag */
        inline bool multiLookEnabled() const { return _multiLookEnabled; }

        /** Compute the avergae frequency shift in range direction between two SLCs*/
        inline double computeRangeFrequencyShift(std::valarray<std::complex<float>> &refAvgSpectrum,
                std::valarray<std::complex<float>> &secAvgSpectrum,
                std::valarray<double> &rangeFrequencies,
                size_t linesPerBlockData,
                size_t fft_size);

        /** estimate the index of the maximum of a vector of data */
        inline void getPeakIndex(std::valarray<float> data,
                                size_t &peakIndex);

        /** Range common band filtering block by block @insar2007product */
        double rangeCommonBandFilter(std::valarray<std::complex<float>> &refSlc,
                std::valarray<std::complex<float>> &secSlc,
                const std::valarray<std::complex<float>> &geometryIfgram,
                const std::valarray<std::complex<float>> &geometryIfgramConj,
                std::valarray<std::complex<float>> &refSpectrum,
                std::valarray<std::complex<float>> &secSpectrum,
                std::valarray<double> &rangeFrequencies,
                isce3::signal::Filter<float> &rngFilter,
                size_t blockRows,
                size_t ncols,
                const size_t maxRangeFilterKernelSize = 256);

        /** Azimuth common band filtering block by block @insar2007product
        //TODO: since there is no windowing is applied in azimuth, no need to revert
        //the windowing effects, and the antenna pattern coefficients are not stored in the
        //SLC yet, will wait for the implementation and revert the attena pattern then
        */
        double azimuthCommonBandFilter(std::valarray<std::complex<float>> &refSlc,
                std::valarray<std::complex<float>> &secSlc,
                const std::valarray<double> &refDoppCentroids,
                const std::valarray<double> &secDoppCentroids,
                std::valarray<std::complex<float>> &refAzimuthSpectrum,
                std::valarray<std::complex<float>> &secAzimuthSpectrum,
                isce3::signal::Filter<float> &azimuthFilter,
                size_t blockRows,
                size_t ncols);

    private:
        void _computeDoppCentroids(const isce3::core::LUT2d<double> & refDoppler,
                                    const isce3::core::LUT2d<double> & secDoppler,
                                    isce3::io::Raster* rngOffsetRaster,
                                    isce3::io::Raster* aziOffsetRaster,
                                    std::valarray<double> &refDopplerCentroids,
                                    std::valarray<double> &secDopplerCentroids,
                                    std::valarray<int> &numOfValidDopplerCentroids);

        int _computeMaxAzimuthFilterKernelSize(const std::valarray<double> &refDopplerCentroids,
                                    const std::valarray<double> &secDopplerCentroids,
                                    const std::valarray<int> &numOfValidDopplerCentroids,
                                    const double bandwidth,
                                    const double prf,
                                    isce3::signal::Filter<float> &aziFilter);

        //Doppler LUT for the refernce SLC
        isce3::core::LUT2d<double> _refDoppler;

        //Doppler LUT for the secondary SLC
        isce3::core::LUT2d<double> _secDoppler;

        // range pixel spacing in meters
        double _rangePixelSpacing = 0.0;

        // starting range shifts between the secondary and reference RSLC in meters
        double _offsetStartingRangeShift = 0.0;

        // reference starting range
        double _refStartRange = 0.0;

        // reference starting azimuth time
        double _refStartAzimuthTime = 0.0;

        // secondary starting range
        double _secStartRange = 0.0;

        // secondary starting azimuth time
        double _secStartAzimuthTime = 0.0;

        // radar wavelength in meters
        double _wavelength = 0.0;

        // number of range looks
        int _rangeLooks = 1;

        // number of azimuth looks
        int _azimuthLooks = 1;

        bool _multiLookEnabled = false;

        // Flag for topo phase removal
        bool _doFlatten = false;

        // Flag for common azimuth band filtering
        bool _doCommonAzimuthBandFilter = false;

        // Azimuth  bandwidth in Hz
        double _azimuthBandwidth = 0.0;

        // Processed azimuth bandwidth after the common band filtering in Hz
        double _processedAzimuthBandwidth = 0.0;

        // Sensor type
        std::string _sensorType = "";

        // Window type
        std::string _windowType = "kaiser";

        // Window parameter for constructing common band filter
        double _windowParameter = 1.6;

        // Flag for common range band filtering
        bool _doCommonRangeBandFilter = false;

        //pulse repetition frequency, in Hz
        double _prf = 0.0;

        // range samping frequency, in Hz
        double _rangeSamplingFrequency = 0.0;

        // range signal bandwidth in Hz
        double _rangeBandwidth = 0.0;

        // Processed range bandwidth after the common band filtering, in Hz
        double _processedRangeBandwidth = 0.0;

        // number of lines per block
        size_t _linesPerBlock = 1024;

        // upsampling factor
        size_t _oversampleFactor = 1;

        // minimum range spectrum overlap fraction between reference and secondary RSLC
        double _minRangeSpectrumOverlapFraction = 0.2;

        // maximum stop band ripple amplitude,
        // in dB below the pass band gain, for the FIR filter
        double _ripple = 40.0;

        // transition width of the common band filter(s),
        // relative to the pass band width (total width of both transition bands)
        double _transitionWidth = 0.15;
};

// Get inline implementations for Crossmul
#define ISCE_SIGNAL_CROSSMUL_ICC
#include "Crossmul.icc"
#undef ISCE_SIGNAL_CROSSMUL_ICC
