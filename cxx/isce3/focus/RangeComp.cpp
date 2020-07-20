#include "RangeComp.h"

#include <algorithm>
#include <limits>

#include <isce3/except/Error.h>

namespace isce { namespace focus {

inline
int getOutputSize(int m, int n, RangeComp::Mode mode)
{
    switch (mode) {
        case RangeComp::Mode::Full  : return m + n - 1;
        case RangeComp::Mode::Valid : return std::max(m, n) - std::min(m, n) + 1;
        case RangeComp::Mode::Same  : return n;
    }

    throw isce::except::RuntimeError(ISCE_SRCINFO(), "unexpected range compression mode");
}

static
std::vector<std::complex<float>>
formRangeReference(const std::vector<std::complex<float>> & chirp, int fftsize)
{
    // initialize reference function with zeros padded to FFT length
    std::vector<std::complex<float>> reffn(fftsize);

    // form matched filter (time-reversed, complex conjugate of chirp)
    auto conj = [](const std::complex<float>& z) { return std::conj(z); };
    std::transform(chirp.rbegin(), chirp.rend(), reffn.begin(), conj);

    // transform to freq domain
    fft::fft1d(reffn.data(), reffn.data(), fftsize);

    return reffn;
}

RangeComp::RangeComp(const std::vector<std::complex<float>> & chirp,
                     int inputsize,
                     int maxbatch,
                     Mode mode)
:
    _chirpsize([=]()
        {
            // make sure chirp size can be cast to int
            std::size_t maxint = std::numeric_limits<int>::max();
            if (chirp.size() > maxint) {
                throw isce::except::OverflowError(ISCE_SRCINFO(), "chirp length exceeds max int");
            }
            return static_cast<int>(chirp.size());
        }()),
    _inputsize([=]()
        {
            if (inputsize < 1) {
                throw isce::except::DomainError(ISCE_SRCINFO(), "number of samples must be > 0");
            }
            return inputsize;
        }()),
    _fftsize(fft::nextFastPower(getOutputSize(_chirpsize, inputsize, Mode::Full))),
    _maxbatch([=]()
        {
            if (maxbatch < 1) {
                throw isce::except::DomainError(ISCE_SRCINFO(), "max batch size must be > 0");
            }
            return maxbatch;
        }()),
    _mode(mode),
    _reffn(formRangeReference(chirp, _fftsize)),
    _wkspc(std::size_t(maxbatch) * _fftsize),
    _fftplan(fft::planfft1d(_wkspc.data(), _wkspc.data(), {maxbatch, _fftsize}, 1)),
    _ifftplan(fft::planifft1d(_wkspc.data(), _wkspc.data(), {maxbatch, _fftsize}, 1))
{}

int RangeComp::outputSize() const
{
    return getOutputSize(chirpSize(), inputSize(), mode());
}

int RangeComp::firstValidSample() const
{
    switch (mode()) {
        case Mode::Full  : return chirpSize() - 1;
        case Mode::Valid : return 0;
        case Mode::Same  : return chirpSize() / 2;
    }

    throw isce::except::RuntimeError(ISCE_SRCINFO(), "unexpected range compression mode");
}

void RangeComp::rangecompress(std::complex<float> * out,
                              const std::complex<float> * in,
                              int batch)
{
    if (batch > maxBatch()) {
        throw isce::except::LengthError(ISCE_SRCINFO(), "batch size exceeds max batch");
    }

    // copy input data to internal workspace buffer & zero pad to FFT length
    int padding = fftSize() - inputSize();
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        const std::complex<float> * src = &in[std::size_t(b) * inputSize()];
        std::complex<float> * dest = &_wkspc[std::size_t(b) * fftSize()];
        std::copy(src, src + inputSize(), dest);
        std::fill_n(dest + inputSize(), padding, std::complex<float>(0.f));
    }

    // FFT convolve
    float scale = 1. / fftSize();
    _fftplan.execute();
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < fftSize(); ++i) {
            _wkspc[std::size_t(b) * fftSize() + i] *= _reffn[i] * scale;
        }
    }
    _ifftplan.execute();

    // crop to output range & copy result to output buffer
    int offset = (mode() == Mode::Full) ? 0 :
                 (mode() == Mode::Valid) ? chirpSize() - 1 :
                 chirpSize() / 2;
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        const std::complex<float> * src = &_wkspc[std::size_t(b) * fftSize()];
        std::complex<float> * dest = &out[std::size_t(b) * outputSize()];
        std::copy_n(src + offset, outputSize(), dest);
    }
}

}}
