#pragma once

namespace isce3 { namespace fft { namespace detail {

void configureFFTLayout(int * n,
                        int * stride,
                        int * dist,
                        int * batch,
                        const int (&dims)[2],
                        int axis);

}}}
