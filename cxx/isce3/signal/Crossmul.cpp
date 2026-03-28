#include "Crossmul.h"

#include "Filter.h"
#include "Looks.h"
#include "Signal.h"

#include <pyre/journal.h>
#include <isce3/fft/FFTUtil.h>

/**
 * Compute the frequency response due to a subpixel shift introduced by
 * upsampling and downsampling

 * @param[in] oversample upsampling factor
 * @param[in] fft_size fft length in range direction
 * @param[in] blockRows number of rows of the block of data
 * @param[out] shiftImpact frequency response (a linear phase) to a sub-pixel
 * shift in time domain introduced by upsampling followed by downsampling
 */
void lookdownShiftImpact(size_t oversample, size_t fft_size, size_t blockRows,
        std::valarray<std::complex<float>> &shiftImpact)
{
    // range frequencies given fft_size and oversampling factor
    std::valarray<double> rangeFrequencies(oversample*fft_size);

    // sampling interval in range
    double dt = 1.0/oversample;

    // get the vector of range frequencies
    isce3::signal::fftfreq(dt, rangeFrequencies);

    // in the process of upsampling the SLCs, creating upsampled interferogram
    // and then looking down the upsampled interferogram to the original size of
    // the SLCs, a shift is introduced in range direction.
    // As an example for a signal with length of 5 and :
    // original sample locations:   0       1       2       3        4
    // upsampled sample locations:  0   0.5 1  1.5  2  2.5  3   3.5  4   4.5
    // Looked down sample locations:  0.25    1.25    2.25    3.25    4.25
    // Obviously the signal after looking down would be shifted by 0.25 pixel in
    // range comared to the original signal. Since a shift in time domain introduces
    // a linear phase in frequency domain, we compute the impact in frequency domain.

    // the constant shift based on the oversampling factor
    double shift = 0.0;
    shift = (1.0 - 1.0/oversample)/2.0;

    // compute the frequency response of the subpixel shift in range direction
    std::valarray<std::complex<float>> shiftImpactLine(oversample*fft_size);
    for (size_t col=0; col<shiftImpactLine.size(); ++col) {
        double phase = -1.0*shift*2.0*M_PI*rangeFrequencies[col];
        shiftImpactLine[col] = std::complex<float> (std::cos(phase),
                                                    std::sin(phase));
    }

    // The impact is the same for each range line. Therefore copying the line
    // for the block
    for (size_t line = 0; line < blockRows; ++line) {
        shiftImpact[std::slice(line*fft_size*oversample, fft_size*oversample, 1)] = shiftImpactLine;
    }
}

// Utility function to get number of OpenMP threads
// (gcc sometimes has problems with omp_get_num_threads)
size_t omp_thread_count() {
    size_t n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

/**
* @param[in, out] refSlc a block of the reference SLC to be filtered
* @param[in, out] secSlc a block of second SLC to be filtered
* @param[in] geometryIfgram a simulated interferogram that contains the geometrical phase due to baseline separation
* @param[in] geometryIfgramConj conjugate of geometryIfgram
* @param[in, out] refSpectrum spectrum of geometryIfgramConj in range direction
* @param[in, out] secSpectrum spectrum of geometryIfgram in range direction
* @param[in] rangeFrequencies frequencies in range direction
* @param[in] rngFilter a filter object
* @param[in] blockLength number of rows
* @param[in] ncols number of columns
* @param[in] maxRangeFilterKernelSize maximum range filter kernel size
* @returns the common bandwidth
*/
double isce3::signal::Crossmul::
rangeCommonBandFilter(std::valarray<std::complex<float>> &refSlc,
                        std::valarray<std::complex<float>> &secSlc,
                        const std::valarray<std::complex<float>> &geometryIfgram,
                        const std::valarray<std::complex<float>> &geometryIfgramConj,
                        std::valarray<std::complex<float>> &refSpectrum,
                        std::valarray<std::complex<float>> &secSpectrum,
                        std::valarray<double> &rangeFrequencies,
                        isce3::signal::Filter<float> &rngFilter,
                        size_t blockLength,
                        size_t ncols,
                        const size_t maxRangeFilterKernelSize)
{
    pyre::journal::debug_t debug("isce.signal.Crossmul.rangeCommonBandFilter");

    // size of the arrays
    size_t vectorLength = refSlc.size();

    // Aligning the spectrum of the two SLCs
    // Shifting the range spectrum of each image according to the local
    // (slope-dependent) wavenumber. This shift in frequency domain is
    // achieved by removing/adding the geometrical phase (representing topography)
    // from/to reference and secondary SLCs in time domain.
    #pragma omp parallel for
    for (size_t i = 0; i < refSlc.size(); i++) {
        refSlc[i] *= geometryIfgramConj[i];
        secSlc[i] *= geometryIfgram[i];
    }


    // determine the frequency shift, in Hz based on the power spectral density of
    // the geometrical interferometric phase using an empirical approach
    double frequencyShift = computeRangeFrequencyShift(refSpectrum,
                                                       secSpectrum,
                                                       rangeFrequencies,
                                                       blockLength,
                                                       ncols);

    debug << "rangeFrequencyShift (MHz): "<< frequencyShift/1e6 << pyre::journal::endl;
    debug << "range bandwidth (MHz): " << _rangeBandwidth/1e6 << pyre::journal::endl;

    // Since the spectrum of the ref and sec SLCs are already aligned,
    // we design the low-pass filter as a band-pass at zero frequency with
    // bandwidth of (range bandwidth - frequency shift)
    const double filterCenterFrequency = 0.0;
    const double filterBandwidth = _rangeBandwidth - fabs(frequencyShift);

    if (_windowType == "kaiser") {
        auto [n, _] = rngFilter._kaiserord(_ripple,
                                          _transitionWidth * filterBandwidth/_rangeSamplingFrequency);
        if (n > maxRangeFilterKernelSize) {
            pyre::journal::error_t err(
                "isce.signal.Crossmul.rangeCommonBandFilter");
            err << "Max filter length exceeded due to insufficient "
                  << "spectral overlap (perpendicular baseline is too large)"
                  << pyre::journal::endl;
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                        "Max filter length exceeded due to insufficient spectral overlap (perpendicular baseline is too large)");
        }
    }

    // Contruct the low pass filter for this block. This filter is
    // common for both SLCs
    rngFilter.constructRangeCommonBandFilter(_rangeSamplingFrequency,
                                    filterCenterFrequency,
                                    filterBandwidth,
                                    ncols,
                                    blockLength,
                                    _windowType,
                                    _windowParameter,
                                    maxRangeFilterKernelSize);

    // low pass filter the ref  slc
    rngFilter.initiateRangeFilter(refSlc,refSpectrum, ncols,  blockLength);
    rngFilter.filter(refSlc, refSpectrum);

    // low pass filter the sec  slc
    rngFilter.initiateRangeFilter(secSlc,secSpectrum, ncols,  blockLength);
    rngFilter.filter(secSlc, secSpectrum);

    // restore the original phase without the geometry phase
    // in case other steps will use the original phase
    #pragma omp parallel for
    for (size_t i = 0; i < refSlc.size(); i++) {
        refSlc[i] *= geometryIfgram[i];
        secSlc[i] *= geometryIfgramConj[i];
    }

    return filterBandwidth;
}

/**
* @param[in, out] refSlc a block of the reference SLC to be filtered
* @param[in, out] secSlc a block of second SLC to be filtered
* @param[in] refDoppCentroids reference doppler centroid
* @param[in] secDoppCentroids secondary doppler centroid
* @param[in, out] refAzimuthSpectrum spectrum of the reference after filtering
* @param[in, out] secAzimuthSpectrum spectrum of the secondary after filtering
* @param[in] azimuthFilter a filter object
* @param[in] blockLength number of rows
* @param[in] ncols number of columns
* @returns the common azimuth bandwidth
*/
double isce3::signal::Crossmul::
azimuthCommonBandFilter(std::valarray<std::complex<float>> &refSlc,
                std::valarray<std::complex<float>> &secSlc,
                const std::valarray<double> &refDoppCentroids,
                const std::valarray<double> &secDoppCentroids,
                std::valarray<std::complex<float>> &refAzimuthSpectrum,
                std::valarray<std::complex<float>> &secAzimuthSpectrum,
                isce3::signal::Filter<float> &azimuthFilter,
                size_t blockRows,
                size_t ncols)
{
    // Construct azimuth common bandpass filter for both reference and secondary
    double processedAzimuthBandwidth = azimuthFilter.constructAzimuthCommonBandFilter(
                refDoppCentroids, secDoppCentroids,
                _azimuthBandwidth,
                _prf, _windowParameter, ncols,
                blockRows, _windowType);

    // Filter a block of data of the reference
    azimuthFilter.initiateAzimuthFilter(refSlc, refAzimuthSpectrum, ncols, blockRows);
    azimuthFilter.filter(refSlc, refAzimuthSpectrum);

    // Filter a block of data of the secondary
    azimuthFilter.initiateAzimuthFilter(secSlc, secAzimuthSpectrum, ncols, blockRows);
    azimuthFilter.filter(secSlc, secAzimuthSpectrum);

    return processedAzimuthBandwidth;
}

/**
* @param[in] refDopplerCentroids doppler centroid frequency for reference, w.r.t slant range
* @param[in] secDopplerCentroids doppler centroid frequency for secondary, w.r.t slant range
* @param[in] numOfDopplerCentroids number of valid doppler centroids
* @param[in] bandwidth intput SLCs bandwidth
* @param[in] prf pulse repeat frequency
* @param[in] beta kaiser window parameter
* @param[in] aziFilter azimuth filter
* @returns the maximum azimuth filter kernel size
*/
int
isce3::signal::Crossmul::_computeMaxAzimuthFilterKernelSize(const std::valarray<double> &refDopplerCentroids,
                                    const std::valarray<double> &secDopplerCentroids,
                                    const  std::valarray<int> &numOfValidDopplerCentroids,
                                    const double bandwidth,
                                    const double prf,
                                    isce3::signal::Filter<float> &aziFilter)
{
    int max_kernel_size = 0;

    // Loop over columns to get the maximum kernel size
   #pragma omp parallel for reduction(max:max_kernel_size)
    for (size_t j = 0; j < refDopplerCentroids.size(); ++j) {
        if (numOfValidDopplerCentroids[j] > 0) {
            double refFreq = refDopplerCentroids[j];
            double secFreq = secDopplerCentroids[j];
            double fshift = std::abs(refFreq - secFreq);

            // The bandwidth should be less than frequency shift
            if (bandwidth < fshift) {
                std::string err_msg = "Bandwidth " +
                                      std::to_string(bandwidth) +
                                      " is less than frequency shift " +
                                      std::to_string(fshift);
                pyre::journal::error_t err(
                    "isce.signal.Crossmul._computeMaxAzimuthFilterKernelSize");
                err << err_msg << pyre::journal::endl;
                throw isce3::except::InvalidArgument(ISCE_SRCINFO(), err_msg);
            }

            // Normalized bandwdith
            const double bw = (bandwidth  - fshift)/ prf;

            // Transition width is specified in terms of output bandwidth, so scale to
            // get width at sample rate of filter.
            const double tw = _transitionWidth * bw;
            if ((bw + tw / 2.0) > 1.0) {
                pyre::journal::error_t err(
                    "isce.signal.Crossmul._maximum_kernel_size");
                err << "Passband + transition cannot exceed Nyquist"
                      << pyre::journal::endl;
                throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                        "Passband + transition cannot exceed Nyquist");
            }

            auto [n, _] = aziFilter._kaiserord(_ripple, tw);
            if (n > max_kernel_size) max_kernel_size = n;
        }
    }

    return max_kernel_size;
}

/**
* @param[in] refDoppler 2d LUT doppler centroid frequency for reference
* @param[in] secDoppler 2d LUT doppler centroid frequency for secondary
* @param[in] rngOffsetRaster range offset product pointer
* @param[out] refDopplerCentroids azimuth mean reference doppler centroids
* @param[out] secDopplerCentroids azimuth mean secondary doppler centroids
* @param[out] numOfValidDopplerCentroids number of valid doppler centroids

*/
void
isce3::signal::Crossmul::_computeDoppCentroids(const isce3::core::LUT2d<double> & refDoppler,
                                    const isce3::core::LUT2d<double> & secDoppler,
                                    isce3::io::Raster* rngOffsetRaster,
                                    isce3::io::Raster* aziOffsetRaster,
                                    std::valarray<double> &refDopplerCentroids,
                                    std::valarray<double> &secDopplerCentroids,
                                    std::valarray<int> &numOfValidDopplerCentroids)
{
    const size_t ncols = rngOffsetRaster->width();
    const size_t nrows = rngOffsetRaster->length();

    if (refDopplerCentroids.size() <= 0) {
        refDopplerCentroids.resize(ncols);
        refDopplerCentroids = 0.0;
    }
    if (secDopplerCentroids.size() <= 0) {
        secDopplerCentroids.resize(ncols);
        secDopplerCentroids = 0.0;
    }
    if (numOfValidDopplerCentroids.size() <= 0) {
        numOfValidDopplerCentroids.resize(ncols);
        numOfValidDopplerCentroids = 0;
    }

    std::valarray<double> rangeOffsets(ncols);
    std::valarray<double> azimuthOffsets(ncols);

    // Compute the doppler centroids for the reference and secondary images
    for (size_t row = 0; row < nrows; row++) {

        rngOffsetRaster->getLine(rangeOffsets, row);
        aziOffsetRaster->getLine(azimuthOffsets, row);

        #pragma omp parallel for
        for (size_t col = 0; col < ncols; col++) {
            // Convert the line/pixel to range doppler coordinates
            double refX = col * rangePixelSpacing() + refStartRange();
            double refY = row / prf() + refStartAzimuthTime();

            double secX = (col + rangeOffsets[col]) * rangePixelSpacing() + secStartRange();
            double secY = (row + azimuthOffsets[col]) / prf() + secStartAzimuthTime();

            // Interpolate the doppler centroids
            if (refDoppler.contains(refY, refX) && secDoppler.contains(secY, secX)) {
                refDopplerCentroids[col] += refDoppler.eval(refY, refX);
                secDopplerCentroids[col] += secDoppler.eval(secY, secX);
                numOfValidDopplerCentroids[col]++;
            }
        }
    }

    // Using the avergae doppler centroid for each column to avoid the artifacts when
    // process block by block
    #pragma omp parallel for
     for (size_t col = 0; col < ncols; col++) {
        if (numOfValidDopplerCentroids[col] > 0) {
            refDopplerCentroids[col] /= numOfValidDopplerCentroids[col];
            secDopplerCentroids[col] /= numOfValidDopplerCentroids[col];
        }
     }
}

void isce3::signal::Crossmul::
crossmul(isce3::io::Raster& refSlcRaster,
        isce3::io::Raster& secSlcRaster,
        isce3::io::Raster& ifgRaster,
        isce3::io::Raster& coherenceRaster,
        isce3::io::Raster* rngOffsetRaster,
        isce3::io::Raster* aziOffsetRaster)
{
    pyre::journal::info_t info("isce.signal.Crossmul.crossmul");
    pyre::journal::debug_t debug("isce.signal.Crossmul.crossmul");

    // number of threads
    size_t nthreads = omp_thread_count();

    // setting local lines per block to avoid modifying class member
    size_t linesPerBlock = _linesPerBlock;

    //filter objects which will be used for azimuth and range common band filtering
    isce3::signal::Filter<float> azimuthFilter;
    isce3::signal::Filter<float> rangeFilter;

    //signal object for refSlc
    isce3::signal::Signal<float> refSignal(nthreads);

    //signal object for secSlc
    isce3::signal::Signal<float> secSignal(nthreads);

    // check consistency of input/output raster shapes
    size_t nrows = refSlcRaster.length();
    size_t ncols = refSlcRaster.width();

    if (ifgRaster.length() != coherenceRaster.length())
        throw isce3::except::LengthError(ISCE_SRCINFO(),
                "interferogram and coherence rasters length do not match");

    if (ifgRaster.width() != coherenceRaster.width())
        throw isce3::except::LengthError(ISCE_SRCINFO(),
                "interferogram and coherence rasters width do not match");

     // maximum range filter kernel size
    size_t maxRangeFilterKernelSize = 0;

    // Compute the range sampling frequency in Hz using the range pixel spacing
    _rangeSamplingFrequency = 1.0 / (_rangePixelSpacing*2.0/isce3::core::speed_of_light);
    if (_doCommonRangeBandFilter) {
        // Determine the max range filter kernel size
        auto [n, _] = rangeFilter._kaiserord(_ripple,
                            _minRangeSpectrumOverlapFraction *
                            rangeBandwidth()/_rangeSamplingFrequency * _transitionWidth);
        maxRangeFilterKernelSize = n;
        debug << "max range filter kernel size: " << maxRangeFilterKernelSize << pyre::journal::endl;
    }

    // Compute FFT size (power of 2) and set up the maximum range window size
    size_t fft_size = isce3::fft::nextFastPower(ncols + maxRangeFilterKernelSize);

    if (fft_size > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "fft_size > INT_MAX");
    if (_oversampleFactor * fft_size > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "_oversampleFactor * fft_size > INT_MAX");

    // force the multilook to be true if azimuth or range looks > 1
    _multiLookEnabled = ((_rangeLooks > 1) || (_azimuthLooks > 1));

    // Declare valarray for range and azimuth doppler centroids used by filter
    std::valarray<double> refDoppCentroids;
    std::valarray<double> secDoppCentroids;
    std::valarray<int> numOfValidDoppCentroids;
    int maxAzimuthFilterKernelSize = 0;
    int overlapSize = 0;
    int halfOverlapSize = 0;

    if (_doCommonAzimuthBandFilter) {
        // Compute the mean doppler centroid of each slant range for
        // the reference and secondary images for each slant range
        refDoppCentroids.resize(fft_size); refDoppCentroids = 0.0;
        secDoppCentroids.resize(fft_size); secDoppCentroids = 0.0;
        numOfValidDoppCentroids.resize(fft_size); numOfValidDoppCentroids = 0;

        info << "determine the max azimuth kernel size" << pyre::journal::newline;
        _computeDoppCentroids(_refDoppler, _secDoppler,
                               rngOffsetRaster,
                               aziOffsetRaster,
                               refDoppCentroids,
                               secDoppCentroids,
                               numOfValidDoppCentroids);

       maxAzimuthFilterKernelSize = _computeMaxAzimuthFilterKernelSize(refDoppCentroids,
                                 secDoppCentroids,
                                 numOfValidDoppCentroids,
                                 _azimuthBandwidth,
                                 _prf,
                                 azimuthFilter);

        debug << "max azimuth window kernel size: " << maxAzimuthFilterKernelSize
              << pyre::journal::endl;
        // to ensure the overlap size is an integer multiple number of azimuth looks.
        int k = (maxAzimuthFilterKernelSize + _azimuthLooks - 1) / _azimuthLooks;

        // Force the kernel size to be even
        k = (k % 2 == 0) ? k : k + 1;

        // The overlap size will be even
        overlapSize = k * _azimuthLooks;
        halfOverlapSize = overlapSize / 2;

        // Compute the lines per block to account for the overlaps between two blocks
        linesPerBlock = ((_linesPerBlock + _azimuthLooks - 1)/ _azimuthLooks) * _azimuthLooks;
        _linesPerBlock = linesPerBlock + overlapSize;

        // The lines per block will be multiple times of the azimuth looks
        linesPerBlock = _linesPerBlock;
    }

    const auto output_rows = ifgRaster.length();
    const auto output_cols = ifgRaster.width();
    if (_multiLookEnabled) {
        // Making sure that the number of rows in each block (linesPerBlock)
        // to be an integer multiple of the number of azimuth looks.
        linesPerBlock = ((_linesPerBlock + _azimuthLooks - 1)/ _azimuthLooks) * _azimuthLooks;

        // checking only multilook interferogram shape is sufficient
        // interferogram and coherence shapes checked to match above
        if (output_rows != nrows / _azimuthLooks)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "multilooked interferogram/coherence raster lengths of unexpected size");

        if (output_cols != ncols / _rangeLooks)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "multilooked interferogram/coherence raster widths of unexpected size");
    } else {
        // checking only multilook interferogram shape is sufficient
        // interferogram and coherence shapes checked to match above
        if (output_rows != nrows)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "full resolution input/output raster lengths do not match");

        if (output_cols != ncols)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "full resolution input/output raster widths do not match");
    }


    //signal object for geometryIfgram
    isce3::signal::Signal<float> geometryIfgramSignal(nthreads);

    //signal object for geometryIfgramConj
    isce3::signal::Signal<float> geometryIfgramConjSignal(nthreads);

    //signal object for refSlc windowing effects revert
    isce3::signal::Signal<float> refWindowSignal(nthreads);

    //signal object for secSlc windowing effects revert
    isce3::signal::Signal<float> secWindowSignal(nthreads);

    // instantiate Looks used for multi-looking the interferogram
    isce3::signal::Looks<float> looksObj;

    const size_t linesPerBlockMLooked = linesPerBlock / _azimuthLooks;
    const size_t ncolsMultiLooked = ncols / _rangeLooks;
    looksObj.nrows(linesPerBlock);
    looksObj.ncols(ncols);
    looksObj.rowsLooks(_azimuthLooks);
    looksObj.colsLooks(_rangeLooks);
    looksObj.nrowsLooked(linesPerBlockMLooked);
    looksObj.ncolsLooked(ncolsMultiLooked);

    // number of blocks to process
    size_t nblocks = nrows / (linesPerBlock - overlapSize);
    if (nblocks == 0) {
        nblocks = 1;
    } else if (nrows % (nblocks * (linesPerBlock - overlapSize)) != 0) {
        nblocks += 1;
    }

    // set up the processed azimuth and range bandwidth
    // if common band filters are enabled, they will be the mean bandwidth of all data block
    // otherwise, they are the original SLC bandwidths
    _processedAzimuthBandwidth = _doCommonAzimuthBandFilter ? 0.0 : _azimuthBandwidth;
    _processedRangeBandwidth = _doCommonRangeBandFilter ? 0.0 : _rangeBandwidth;

    // size of not-unsampled valarray
    const auto spectrumSize = fft_size * linesPerBlock;

    // size of unsampled valarray
    const auto spectrumUpsampleSize = _oversampleFactor * spectrumSize;

    // storage for a block of reference SLC data
    std::valarray<std::complex<float>> refSlc(spectrumSize);

    // storage for a block of secondary SLC data
    std::valarray<std::complex<float>> secSlc(spectrumSize);

    // storage for a block of range offsets
    std::valarray<double> rngOffset(spectrumSize);

    // InSAR phase due to topography
    std::valarray<std::complex<float>> geometryIfgram;

    // complex conjugate of geometryIfgram
    // both the flattening and range common band filtering will use
    // this variable, so creating a buffer here
    std::valarray<std::complex<float>> geometryIfgramConj(spectrumSize);

    // upsampled interferogram
    std::valarray<std::complex<float>> ifgramUpsampled(_oversampleFactor*ncols*linesPerBlock);

    // full resolution interferogram
    std::valarray<std::complex<float>> ifgram(ncols*linesPerBlock);

    // multi-looked interferogram
    std::valarray<std::complex<float>> ifgramMultiLooked;

    // multi-looked power of reference SLC
    std::valarray<float> refPowerLooked;

    // multi-looked power of secondary SLC
    std::valarray<float> secPowerLooked;

    // coherence for multi-looked and full-res interferogram
    std::valarray<float> coherence;

    if (_multiLookEnabled) {
        // resize following valarrays from empty
        const auto mlookSize = ncolsMultiLooked*linesPerBlockMLooked;
        ifgramMultiLooked.resize(mlookSize);
        coherence.resize(mlookSize);
        refPowerLooked.resize(mlookSize);
        secPowerLooked.resize(mlookSize);
    }
    else {
        coherence.resize(ncols*linesPerBlock);
    }

    // storage for spectrum of the block of data in reference SLC
    std::valarray<std::complex<float>> refSpectrum;

    // storage for spectrum of the block of data in secondary SLC
    std::valarray<std::complex<float>> secSpectrum;

    // upsampled spectrum of the block of reference SLC
    std::valarray<std::complex<float>> refSpectrumUpsampled;

    // upsampled spectrum of the block of secondary SLC
    std::valarray<std::complex<float>> secSpectrumUpsampled;

    // upsampled block of reference SLC
    std::valarray<std::complex<float>> refSlcUpsampled;

    // upsampled block of secondary SLC
    std::valarray<std::complex<float>> secSlcUpsampled;

    if ((_oversampleFactor > 1) || _doCommonRangeBandFilter) {
        refSpectrum.resize(spectrumSize);
        secSpectrum.resize(spectrumSize);
    }

    // only resize valarrays and init FFT when oversampling
    if (_oversampleFactor > 1) {

        refSpectrumUpsampled.resize(spectrumUpsampleSize);
        secSpectrumUpsampled.resize(spectrumUpsampleSize);
        refSlcUpsampled.resize(spectrumUpsampleSize);
        secSlcUpsampled.resize(spectrumUpsampleSize);

        // make forward and inverse fft plans for the reference SLC
        refSignal.forwardRangeFFT(refSlc, refSpectrum, fft_size, linesPerBlock);
        refSignal.inverseRangeFFT(refSpectrumUpsampled, refSlcUpsampled,
                fft_size*_oversampleFactor, linesPerBlock);

        // make forward and inverse fft plans for the secondary SLC
        secSignal.forwardRangeFFT(secSlc, secSpectrum, fft_size, linesPerBlock);
        secSignal.inverseRangeFFT(secSpectrumUpsampled, secSlcUpsampled,
                fft_size*_oversampleFactor, linesPerBlock);
    }

    // looking down the upsampled interferogram may shift the samples by a fraction of a pixel
    // depending on the oversample factor. predicting the impact of the shift in frequency domain
    // which is a linear phase allows to account for it during the upsampling process
    std::valarray<std::complex<float>> shiftImpact(spectrumUpsampleSize);
    lookdownShiftImpact(_oversampleFactor,  fft_size,
                        linesPerBlock, shiftImpact);

    // Declare valarray for range frequencies used by filter
    std::valarray<double> rangeFrequencies;

    // Declare valarray for azimuth spectrum used by filter
    std::valarray<std::complex<float>> refAzimuthSpectrum;
    std::valarray<std::complex<float>> secAzimuthSpectrum;

    // Retrieve the original filter to revert the range windowing effects
    std::valarray<std::complex<float>> originalRangeFilter;

    if (_doCommonAzimuthBandFilter) {
        // Allocate storage for azimuth spectrum
        refAzimuthSpectrum.resize(spectrumSize);
        secAzimuthSpectrum.resize(spectrumSize);
    }

    if (_doCommonRangeBandFilter) {
        geometryIfgram.resize(spectrumSize);
        rangeFrequencies.resize(fft_size);

        // Compute the range frequency for each pixel
        fftfreq(1.0/_rangeSamplingFrequency, rangeFrequencies);

        // For NISAR, use a standard Kaiser window in frequency domain to
        // compensate the windowing effects in range direction
        // NOTE: This function is sensor dependent
        if (_sensorType == "NISAR") {
            originalRangeFilter.resize(fft_size);
            std::valarray<double> subBandCenterFrequencies{0.0};
            std::valarray<double> subBandBandwidths{_rangeSamplingFrequency};
            rangeFilter.constructRangeBandpassKaiser(subBandCenterFrequencies,
                                                    subBandBandwidths,
                                                    1.0/_rangeSamplingFrequency,
                                                    fft_size, rangeFrequencies, 1.6,
                                                    originalRangeFilter);
        }

        // Construct the FFTW plan for both geometryIfgram and its conjugation
        geometryIfgramSignal.forwardRangeFFT(geometryIfgram, refSpectrum,
                        fft_size, linesPerBlock);
        geometryIfgramSignal.inverseRangeFFT(refSpectrum, geometryIfgram,
                        fft_size, linesPerBlock);

        geometryIfgramConjSignal.forwardRangeFFT(geometryIfgramConj, secSpectrum,
                       fft_size, linesPerBlock);
        geometryIfgramConjSignal.inverseRangeFFT(secSpectrum, geometryIfgramConj,
                      fft_size, linesPerBlock);

        refWindowSignal.forwardRangeFFT(refSlc, refSpectrum,
                        fft_size, linesPerBlock);
        refWindowSignal.inverseRangeFFT(refSpectrum, refSlc,
                        fft_size, linesPerBlock);

        secWindowSignal.forwardRangeFFT(secSlc, secSpectrum,
                        fft_size, linesPerBlock);
        secWindowSignal.inverseRangeFFT(secSpectrum, secSlc,
                        fft_size, linesPerBlock);
    }

    // loop over all blocks
    std::cout << "nblocks: " << nblocks << std::endl;
    info << "nblocks: " << nblocks <<  pyre::journal::newline;
    for (size_t block = 0; block < nblocks; ++block) {
        std::cout << "block: " << block << std::endl;
        info << "block: " << block <<  pyre::journal::newline;
        // start row for this block
        const auto rowStart = block * (linesPerBlock - overlapSize);

        //number of lines of data in this block. blockRowsData<= linesPerBlock
        //Note that linesPerBlock is fixed number of lines
        //blockRowsData might be less than or equal to linesPerBlock.
        //e.g. if nrows = 512, and linesPerBlock = 100, then
        //blockRowsData for last block will be 12
        const auto blockRowsData = std::min(nrows - rowStart, linesPerBlock);

        // fill the valarray with zero before getting the block of the data
        refSlc = 0;
        secSlc = 0;
        ifgramUpsampled = 0;
        ifgram = 0;
        geometryIfgramConj = 0;
        geometryIfgram = 0;
        rngOffset = 0;

        // get a block of reference and secondary SLC data
        // and a block of range offsets
        // This will change once we have the functionality to
        // get a block of data directly in to a slice
        // This zero-pads SLCs in range
        std::valarray<std::complex<float>> dataLine(ncols);
        for (size_t line = 0; line < blockRowsData; ++line) {
            refSlcRaster.getLine(dataLine, rowStart + line);
            refSlc[std::slice(line*fft_size, ncols, 1)] = dataLine;
            secSlcRaster.getLine(dataLine, rowStart + line);
            secSlc[std::slice(line*fft_size, ncols, 1)] = dataLine;
        }

        // load the range offsets that are required by flattening,
        // common range filters
        if (_doFlatten || _doCommonRangeBandFilter) {
            std::valarray<double> offsetLine(ncols);
            for (size_t line = 0; line < blockRowsData; ++line) {
                rngOffsetRaster->getLine(offsetLine, rowStart + line);
                rngOffset[std::slice(line*fft_size, ncols, 1)] = offsetLine;
            }
        }

        // common range band-pass filtering
        if (_doCommonRangeBandFilter) {
            // Some diagnostic messages to make sure everything has been configured
            debug << "range pixel spacing (m): " << _rangePixelSpacing << pyre::journal::endl;
            debug << "wavelength (m): " << _wavelength << pyre::journal::endl;

            refWindowSignal.forward(refSlc, refSpectrum);
            secWindowSignal.forward(secSlc, secSpectrum);

            // Convert range offset from meters to complex one-way geometric phase
            #pragma omp parallel for
            for (size_t line = 0; line < blockRowsData; ++line) {
                for (size_t col = 0; col < ncols; ++col) {

                    // one-way geometric phase to shift the spectrum
                    double phase = 2.0 * M_PI
                        * _rangePixelSpacing*rngOffset[line*fft_size+col]
                        / _wavelength;

                    geometryIfgram[line*fft_size + col] =
                        std::complex<float> (std::cos(phase), std::sin(phase));
                    geometryIfgramConj[line*fft_size + col] = std::conj(geometryIfgram[line*fft_size + col]);

                    // revert the original windowing effects along the slant range direction
                    if (originalRangeFilter.size() > 0) {
                        refSpectrum[line*fft_size + col] /= originalRangeFilter[col];
                        secSpectrum[line*fft_size + col] /= originalRangeFilter[col];
                    }
                }
            }

            refWindowSignal.inverse(refSpectrum, refSlc);
            secWindowSignal.inverse(secSpectrum, secSlc);

            // Forward FFT to compute geometry-dependent spectrum to determine the
            // frequency shift
            geometryIfgramConjSignal.forward(geometryIfgramConj, refSpectrum);
            geometryIfgramSignal.forward(geometryIfgram, secSpectrum);

            // Apply range common band filter to ref and sec SLC
            // and the resultant refSlc and secSlc have no topo phase removal
            _processedRangeBandwidth += rangeCommonBandFilter(refSlc, secSlc, geometryIfgram,
                        geometryIfgramConj, refSpectrum, secSpectrum,
                        rangeFrequencies, rangeFilter, linesPerBlock, fft_size,
                        maxRangeFilterKernelSize);
        }

        // Apply the azimuth common band-pass filter to the reference and secondary SLCs
        if (_doCommonAzimuthBandFilter) {
             _processedAzimuthBandwidth += azimuthCommonBandFilter(refSlc, secSlc,
                                    refDoppCentroids, secDoppCentroids,
                                    refAzimuthSpectrum, secAzimuthSpectrum,
                                    azimuthFilter, linesPerBlock, fft_size);
        }

        // upsample the reference and secondary SLCs
        if (_oversampleFactor == 1) {
            refSlcUpsampled = refSlc;
            secSlcUpsampled = secSlc;
        } else {
            refSignal.upsample(refSlc, refSlcUpsampled, linesPerBlock, fft_size,
                               _oversampleFactor, shiftImpact);
            secSignal.upsample(secSlc, secSlcUpsampled, linesPerBlock, fft_size,
                               _oversampleFactor, shiftImpact);
        }

        // Compute oversampled interferogram data
        #pragma omp parallel for
        for (size_t line = 0; line < blockRowsData; line++) {
            for (size_t col = 0; col < _oversampleFactor*ncols; col++) {
                ifgramUpsampled[line*(_oversampleFactor*ncols) + col] =
                        refSlcUpsampled[line*(_oversampleFactor*fft_size) + col]*
                        std::conj(secSlcUpsampled[line*(_oversampleFactor*fft_size) + col]);
            }
        }

        if (_doFlatten) {
            // Read range offsets
            std::valarray<double> offsetLine(ncols);
            for (size_t line = 0; line < blockRowsData; ++line) {
                rngOffsetRaster->getLine(offsetLine, rowStart + line);
                rngOffset[std::slice(line*ncols, ncols, 1)] = offsetLine + _offsetStartingRangeShift / _rangePixelSpacing;
            }

            #pragma omp parallel for
            for (size_t line = 0; line < blockRowsData; ++line) {
                for (size_t col = 0; col < ncols; ++col) {
                    double phase = 4.0*M_PI*_rangePixelSpacing*rngOffset[line*fft_size+col]/_wavelength;
                    geometryIfgramConj[line*fft_size + col] = std::complex<float> (std::cos(phase),
                                                                            -1.0*std::sin(phase));
                }
            }
        }

        // Reclaim the extra oversample looks across
        float ov = _oversampleFactor;
        #pragma omp parallel for
        for (size_t line = 0; line < blockRowsData; line++) {
            for (size_t col = 0; col < ncols; col++) {
                std::complex<float> sum = 0;
                for (size_t j=0; j< _oversampleFactor; j++)
                    sum += ifgramUpsampled[line*(ncols*_oversampleFactor) + j + col*_oversampleFactor];
                ifgram[line*ncols + col] = sum/ov;

                if (_doFlatten) {
                    ifgram[line*ncols + col] *= geometryIfgramConj[line*fft_size + col];
                }
            }
        }

        // Take looks down (summing columns)
        if (_multiLookEnabled) {
            // mulitlook interferogram and set raster
            looksObj.ncols(ncols);
            looksObj.colsLooks(_rangeLooks);
            looksObj.multilook(ifgram, ifgramMultiLooked);

            size_t rowNewStart = 0;
            // if there is only one block, the nrowsMultiLooked will be equal to blockRowsData
            size_t nrowsMultiLooked = (nblocks == 1) ? blockRowsData : blockRowsData - halfOverlapSize;
            nrowsMultiLooked /= _azimuthLooks;

            // The first block
            std::slice dataSlice = std::slice(0, ncolsMultiLooked * nrowsMultiLooked, 1);
            if (block > 0) {
                rowNewStart = (rowStart + halfOverlapSize)/_azimuthLooks;
                if (block != (nblocks - 1)) {
                    nrowsMultiLooked = (blockRowsData - overlapSize) / _azimuthLooks;
                    dataSlice = std::slice(halfOverlapSize/_azimuthLooks * ncolsMultiLooked,
                                    ncolsMultiLooked * nrowsMultiLooked, 1);
                } else { // The last block
                    nrowsMultiLooked = (blockRowsData - halfOverlapSize) / _azimuthLooks;
                    dataSlice = std::slice(halfOverlapSize/_azimuthLooks * ncolsMultiLooked,
                                    ncolsMultiLooked * nrowsMultiLooked, 1);
                }
            }
             // set the block of interferogram
            std::valarray<std::complex<float>> ifgramSlice= ifgramMultiLooked[dataSlice];
            ifgRaster.setBlock(ifgramSlice, 0, rowNewStart,
                               ncolsMultiLooked, nrowsMultiLooked);

            // multilook SLC to power for coherence computation
            // refPowerLooked = average(abs(refSlc)^2)
            if (_oversampleFactor == 1) {
                looksObj.ncols(fft_size);
                looksObj.multilook(refSlc, refPowerLooked, 2);
                looksObj.multilook(secSlc, secPowerLooked, 2);
            } else {
                // update looksObj so SlcUpsampled can be mulitlooked
                looksObj.ncols(_oversampleFactor*fft_size);
                looksObj.colsLooks(_oversampleFactor*_rangeLooks);
                looksObj.multilook(refSlcUpsampled, refPowerLooked, 2);
                looksObj.multilook(secSlcUpsampled, secPowerLooked, 2);
            }

            // compute coherence
            #pragma omp parallel for
            for (size_t i = 0; i< ifgramMultiLooked.size(); ++i) {
                coherence[i] = std::abs(ifgramMultiLooked[i])/
                        std::sqrt(refPowerLooked[i]*secPowerLooked[i]);
            }

            // set coherence raster
            std::valarray<float> coherenceSlice = coherence[dataSlice];
            coherenceRaster.setBlock(coherenceSlice, 0, rowNewStart,
                                    ncolsMultiLooked,nrowsMultiLooked);
        } else {
            // The first block
            size_t rowNewStart = 0;
            // if there is only one block, the nrowsValid will be equal to blockRowsData
            size_t nrowsValid = (nblocks == 1) ? blockRowsData : blockRowsData - halfOverlapSize;
            std::slice dataSlice = std::slice(0, nrowsValid * ncols, 1);
            if (block > 0) {
                rowNewStart = rowStart + halfOverlapSize;
                if (block != (nblocks - 1)) {
                    nrowsValid = blockRowsData - overlapSize;
                    dataSlice = std::slice(halfOverlapSize * ncols, nrowsValid, 1);
                } else { // the last block
                    nrowsValid = blockRowsData - halfOverlapSize;
                    dataSlice = std::slice(halfOverlapSize * ncols,nrowsValid, 1);
                }
            }

            std::valarray<std::complex<float>> ifgramSlice= ifgram[dataSlice];
            // set the block of interferogram
            ifgRaster.setBlock(ifgramSlice, 0, rowNewStart, ncols, nrowsValid);

            // fill coherence with ones (no need to compute result)
            coherence = 1.0;
            // set the block of coherence
            std::valarray<float> coherenceSlice = coherence[dataSlice];
            coherenceRaster.setBlock(coherenceSlice, 0, rowNewStart, ncols,
                                     nrowsValid);
        }
    }

    // update the azimuth and range bandwidth after common band filtering
    // using the mean bandwidth
    if (_doCommonRangeBandFilter) _processedRangeBandwidth /= nblocks;
    if (_doCommonAzimuthBandFilter) _processedAzimuthBandwidth /= nblocks;
}
