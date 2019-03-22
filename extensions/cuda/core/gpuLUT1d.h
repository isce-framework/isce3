//
// Author: Bryan Riel
// Copyright 2017-2018
//

#ifndef ISCE_CUDA_CORE_GPULUT1D_H
#define ISCE_CUDA_CORE_GPULUT1D_H

#include <cmath>
#include "isce/core/LUT1d.h"

#include "Common.h"

// Declaration
namespace isce {
    namespace cuda {
        namespace core {
            template <typename T> class gpuLUT1d;
        }
    }
}

// gpuLUT1d declaration
template <typename T>
class isce::cuda::core::gpuLUT1d {

    public:
        // Disallow default constructor
        CUDA_HOSTDEV gpuLUT1d() = delete;
    
        /** Deep copy constructor from CPU LUT1d */
        CUDA_HOST gpuLUT1d(const isce::core::LUT1d<T> &);

        /** Shallow copy constructor on device */
        CUDA_HOSTDEV gpuLUT1d(gpuLUT1d<T> &);

        /** Shallow assignment operator on device */
        CUDA_HOSTDEV gpuLUT1d & operator=(gpuLUT1d<T> &);

        /** Destructor */
        ~gpuLUT1d();

        /** Access to coordinates */
        CUDA_HOSTDEV inline double * coords() { return _coords; }

        /** Read-only access to coordinates */
        CUDA_HOSTDEV inline const double * coords() const { return _coords; }

        /** Set the coordinates */
        CUDA_HOSTDEV inline void coords(double * c) { _coords = c; }

        /** Access to values */
        CUDA_HOSTDEV inline T * values() { return _values; }

        /** Read-only access to values */
        CUDA_HOSTDEV inline const T * values() const { return _values; }

        /** Set the values */
        CUDA_HOSTDEV inline void values(T * v) { _values = v; }

        /** Get extrapolate flag */
        CUDA_HOSTDEV inline bool extrapolate() const { return _extrapolate; }

        /** Set extrapolation flag */
        CUDA_HOSTDEV inline void extrapolate(bool flag) { _extrapolate = flag; }

        /** Get flag for having data */
        CUDA_HOSTDEV inline bool haveData() const { return _haveData; }

        /** Get reference value */
        CUDA_HOSTDEV inline T refValue() const { return _refValue; }

        /** Get size info of LUT */
        CUDA_HOSTDEV inline size_t size() const { return _size; }

        /** Set size info of LUT */
        CUDA_HOSTDEV inline void size(size_t s) { _size = s; }

        /** Evaluate the LUT */
        CUDA_DEV T eval(double x) const;

        /** Evaluate the LUT from host (test function) */
        CUDA_HOST T eval_h(double x);

    // Data members
    private:
        bool _haveData;
        T _refValue;
        double * _coords;
        T * _values;
        size_t _size;
        bool _extrapolate;
        bool _owner;
};

#endif

// end of file
