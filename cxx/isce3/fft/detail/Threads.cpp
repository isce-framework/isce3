#include "Threads.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace isce3 { namespace fft { namespace detail {

int getMaxThreads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}}}
