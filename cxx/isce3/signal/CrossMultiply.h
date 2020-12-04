#pragma once

#include <isce3/signal/forward.h>

#include <complex>

#include <isce3/core/EMatrix.h>
#include <isce3/signal/Signal.h>

namespace isce3 { namespace signal {

/** Interferogram processor */
class CrossMultiply {
public:
    /**
     * Constructor
     *
     * Forms an interferogram by cross multiplication of two coregistered
     * Single Look Complex (SLC) images.
     *
     * \param[in] nrows     Number of rows of the block of the input data
     * \param[in] ncols     Number of columns of the block of the input data
     * \param[in] upsample  Upsampling factor (by default = 2)
     */
    CrossMultiply(int nrows, int ncols, int upsample = 2);

    int nrows() const { return _nrows; }

    int ncols() const { return _ncols; }

    int upsample() const { return _upsample; }

    int fftsize() const { return _fftsize; }

    /**
     * Perform interferogram formation on a block of input data
     *
     * Computes the upsampled reference and secondary SLCs,
     * cross multiplies the upsampled SLCs and looks down by the
     * upsampling factor to generate the full resolution interferogram.
     *
     * \param[out] out_ifgram    Full resolution output interferogram
     * \param[in]  ref_slc       Reference SLC
     * \param[in]  sec_slc       Secondary SLC
     */
    void
    crossmultiply(isce3::core::EArray2D<std::complex<float>>& out_ifgram,
                  const isce3::core::EArray2D<std::complex<float>>& ref_slc,
                  const isce3::core::EArray2D<std::complex<float>>& sec_slc);

private:
    int _nrows;
    int _ncols;
    int _upsample;
    int _fftsize;

    isce3::core::EArray2D<std::complex<float>> _ref_slc;
    isce3::core::EArray2D<std::complex<float>> _sec_slc;
    isce3::core::EArray2D<std::complex<float>> _ref_slc_spec;
    isce3::core::EArray2D<std::complex<float>> _sec_slc_spec;
    isce3::core::EArray2D<std::complex<float>> _ref_slc_up;
    isce3::core::EArray2D<std::complex<float>> _sec_slc_up;
    isce3::core::EArray2D<std::complex<float>> _ref_slc_up_spec;
    isce3::core::EArray2D<std::complex<float>> _sec_slc_up_spec;
    isce3::core::EArray2D<std::complex<float>> _ifgram_up;

    isce3::signal::Signal<float> _signal;
};

}} // namespace isce3::signal
