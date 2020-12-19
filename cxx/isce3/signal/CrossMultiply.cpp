#include "CrossMultiply.h"

#include <isce3/core/EMatrix.h>
#include <isce3/except/Error.h>
#include <isce3/fft/FFT.h>
#include <isce3/signal/multilook.h>

namespace isce3 { namespace signal {

CrossMultiply::CrossMultiply(int nrows, int ncols, int upsample)
    : _nrows([=]() {
          if (nrows < 1) {
              throw isce3::except::DomainError(ISCE_SRCINFO(),
                                               "number of rows must be > 0");
          }
          return nrows;
      }()),
      _ncols([=]() {
          if (ncols < 1) {
              throw isce3::except::DomainError(ISCE_SRCINFO(),
                                               "number of rows columns be > 0");
          }
          return ncols;
      }()),
      _upsampleFactor([=]() {
          if (upsample < 1) {
              throw isce3::except::DomainError(ISCE_SRCINFO(),
                                               "upsampling factor must be > 0");
          }
          return upsample;
      }()),
      _fftsize(isce3::fft::nextFastPower(ncols)), _ref_slc(_nrows, _fftsize),
      _sec_slc(_nrows, _fftsize), _ref_slc_spec(_nrows, _fftsize),
      _sec_slc_spec(_nrows, _fftsize),
      _ref_slc_up(_nrows, _fftsize * _upsampleFactor),
      _sec_slc_up(_nrows, _fftsize * _upsampleFactor),
      _ref_slc_up_spec(_nrows, _fftsize * _upsampleFactor),
      _sec_slc_up_spec(_nrows, _fftsize * _upsampleFactor),
      _ifgram_up(_nrows, _fftsize * _upsampleFactor)
{
    // make forward and inverse fft plans
    _signal.forwardRangeFFT(_ref_slc.data(), _ref_slc_spec.data(), _fftsize,
                            _nrows);
    _signal.inverseRangeFFT(_ref_slc_up_spec.data(), _ref_slc_up.data(),
                            _fftsize * _upsampleFactor, _nrows);
}

void CrossMultiply::crossmultiply(
        Eigen::Ref<isce3::core::EArray2D<std::complex<float>>> ifgram,
        const Eigen::Ref<const isce3::core::EArray2D<std::complex<float>>>&
                ref_slc,
        const Eigen::Ref<const isce3::core::EArray2D<std::complex<float>>>&
                sec_slc)
{
    // sanity checks
    if (ref_slc.rows() != _nrows) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Number of rows in reference slc must "
                                         "be the same as the instance nrows.");
    }
    if (sec_slc.rows() != _nrows) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Number of rows in secondary slc must "
                                         "be the same as the instance nrows.");
    }
    if (ref_slc.cols() != _ncols) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Number of columns in reference slc must be "
                                "the same as the instance ncols.");
    }
    if (sec_slc.cols() != _ncols) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Number of columns in secondary slc must be "
                                "the same as the instance ncols.");
    }
    if (ifgram.rows() != _nrows) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Number of rows in output interferogram must "
                                "be the same as the instance nrows.");
    }
    if (ifgram.cols() != _ncols) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Number of columns in output interferogram "
                                "must be the same as the instance ncols.");
    }
    if (_upsampleFactor == 1) {

        ifgram = ref_slc * sec_slc.conjugate();

    } else {

        // copy the input data to internal container which is zero padded to
        // fftsize
        _ref_slc.block(0, 0, _nrows, _ncols) = ref_slc;
        _sec_slc.block(0, 0, _nrows, _ncols) = sec_slc;

        // upsample the SLCs
        _signal.upsample(_ref_slc, _ref_slc_up);
        _signal.upsample(_sec_slc, _sec_slc_up);

        // create the upsampled interferogram
        _ifgram_up = _ref_slc_up * _sec_slc_up.conjugate();

        // look down by the upsample factor in range
        ifgram = isce3::signal::multilookSummed(
                _ifgram_up.block(0, 0, _nrows, _ncols * _upsampleFactor), 1,
                _upsampleFactor);
    }
}

}} // namespace isce3::signal
