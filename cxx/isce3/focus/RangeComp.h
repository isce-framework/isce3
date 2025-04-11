#pragma once

#include <complex>
#include <vector>

#include <isce3/fft/FFT.h>

namespace isce3 { namespace focus {

/** Range compression processor */
class RangeComp {
public:
    /** Convolution output mode */
    enum class Mode {
        /**
         * The output contains the full discrete convolution of the input with
         * the matched filter.
         *
         * For an input signal and chirp of size N and M, the output size is
         * (N + M - 1).
         */
        Full,

        /**
         * The output contains only the valid discrete convolution of the input
         * with the matched filter.
         *
         * For an input signal and chirp of size N and M, the output size is
         * (max(M, N) - min(M, N) + 1).
         */
        Valid,

        /**
         * The output contains the discrete convolution of the input with the
         * matched filter, cropped to the same size as the input signal.
         */
        Same
    };

    /**
     * Constructor
     *
     * Forms a matched filter from the time-reversed complex conjugate of the
     * chirp replica and creates FFT plans for frequency domain convolution
     * with the matched filter.
     *
     * \param[in] chirp     Time-domain replica of the transmitted chirp waveform
     * \param[in] inputsize Number of range samples in the signal to be compressed
     * \param[in] maxbatch  Max batch size
     * \param[in] mode      Convolution output mode
     */
    RangeComp(const std::vector<std::complex<float>> & chirp,
              int inputsize,
              int maxbatch = 1,
              Mode mode = Mode::Full);

    /** Number of samples in chirp */
    int chirpSize() const { return _chirpsize; }

    /** Expected number of samples in the input signal to be compressed */
    int inputSize() const { return _inputsize; }

    /** FFT length */
    int fftSize() const { return _fftsize; }

    /** Max batch size */
    int maxBatch() const { return _maxbatch; }

    /** Output mode */
    Mode mode() const { return _mode; }

    /** Output number of samples */
    int outputSize() const;

    /**
     * Return the (zero-based) index of the first fully-focused pixel in the
     * output.
     */
    int firstValidSample() const;

    /**
     * Apply a notch to the baseband frequency-domain representation of the
     * chirp reference function.
     * 
     * \param[in] frequency Center frequency of the notch normalized by the
     *                      sample rate, so should be in interval [-0.5, 0.5)
     * \param[in] bandwidth Total width for the notch taper.  If zero, then
     *                      only the nearest FFT bin will be set to zero.
     */
    void applyNotch(double frequency, double bandwidth = 0.0);

    /**
     * Get a pointer to the frequency spectrum of the chirp that will be used
     * for range compression.  Its length is equal to fftSize().
     */
    const std::complex<float>* chirp_spectrum() const {
        return _reffn.data();
    }

    /**
     * Perform pulse compression on a batch of input signals
     *
     * Computes the frequency domain convolution of the input with the reference
     * function.
     *
     * \throws LengthError  If \p batch exceeds the max batch size
     *
     * \param[out] out      Range-compressed data
     * \param[in]  in       Input data
     * \param[in]  batch    Input batch size
     */
    void rangecompress(std::complex<float> * out, const std::complex<float> * in, int batch = 1);

private:
    int _chirpsize;
    int _inputsize;
    int _fftsize;
    int _maxbatch;
    Mode _mode;
    std::vector<std::complex<float>> _reffn;
    std::vector<std::complex<float>> _wkspc;
    isce3::fft::FwdFFTPlan<float> _fftplan;
    isce3::fft::InvFFTPlan<float> _ifftplan;
};

}}
