#include "ConfigureFFTLayout.h"

#include <string>

#include <isce/except/Error.h>

namespace isce { namespace fft { namespace detail {

void configureFFTLayout(int * n,
                        int * stride,
                        int * dist,
                        int * batch,
                        const int (&dims)[2],
                        int axis)
{
    // check for out-of-range axis
    if (axis < -2 || axis >= 2) {
        std::string errmsg = "axis (" + std::to_string(axis) + ") out of range for 2-D array";
        throw isce::except::OutOfRange(ISCE_SRCINFO(), errmsg);
    }

    // wrap around negative axis
    if (axis < 0) { axis = 2 - axis; }

    // configure FFTW advanced layout params
    if (axis == 0) {
        // column-wise FFT (assuming row-major data)
        *n = dims[0];
        *stride = dims[1];
        *dist = 1;
        *batch = dims[1];
    }
    else { // axis == 1
        // row-wise FFT (assuming row-major data)
        *n = dims[1];
        *stride = 1;
        *dist = dims[1];
        *batch = dims[0];
    }
}

}}}
