//
// Author: Bryan Riel
// Copyright 2017-2019
//

#ifndef ISCE_CUDA_CORE_GPULUT2D_H
#define ISCE_CUDA_CORE_GPULUT2D_H

#include <cmath>
#include <isce/core/LUT2d.h>
#include <isce/cuda/core/Common.h>
#include <isce/cuda/core/gpuInterpolator.h>

// Declaration
namespace isce {
    namespace cuda {
        namespace core {
            template <typename T> class gpuLUT2d;
        }
    }
}

// gpuLUT2d declaration
template <typename T>
class isce::cuda::core::gpuLUT2d {

    public:
        // Disallow default constructor
        CUDA_HOSTDEV gpuLUT2d() = delete;
    
        /** Deep copy constructor from CPU LUT1d */
        CUDA_HOST gpuLUT2d(const isce::core::LUT2d<T> &);

        /** Shallow copy constructor on device */
        CUDA_HOSTDEV gpuLUT2d(gpuLUT2d<T> &);

        /** Shallow assignment operator on device */
        CUDA_HOSTDEV gpuLUT2d & operator=(gpuLUT2d<T> &);

        /** Destructor */
        ~gpuLUT2d();

        /** Initialize interpolation object on device. */
        CUDA_HOST void initInterp();

        /** Finalize/delete interpolation object on device. */
        CUDA_HOST void finalizeInterp();

        /** Get starting X-coordinate */
        CUDA_HOSTDEV inline double xStart() const { return _xstart; }

        /** Get starting Y-coordinate */
        CUDA_HOSTDEV inline double yStart() const { return _ystart; }

        /** Get X-spacing */
        CUDA_HOSTDEV inline double xSpacing() const { return _dx; }

        /** Get Y-spacing */
        CUDA_HOSTDEV inline double ySpacing() const { return _dy; }

        /** Get LUT length (number of lines) */
        CUDA_HOSTDEV inline size_t length() const { return _length; }

        /** Get LUT width (number of samples) */
        CUDA_HOSTDEV inline size_t width() const { return _width; }

        /** Get the reference value */
        CUDA_HOSTDEV inline T refValue() const { return _refValue; }

        /** Get flag for having data */
        CUDA_HOSTDEV inline bool haveData() const { return _haveData; }

        /** Get bounds error flag */
        CUDA_HOSTDEV inline bool boundsError() const { return _boundsError; }

        /** Get pointer to interpolator */
        CUDA_HOSTDEV inline isce::cuda::core::gpuInterpolator<T> ** interp() const {
            return _interp;
        }

        /** Access to data values */
        CUDA_HOSTDEV inline T * data() { return _data; }

        /** Read-only access to data values */
        CUDA_HOSTDEV inline const T * data() const { return _data; }

        /** Set the data values pointer */
        CUDA_HOSTDEV inline void data(T * v) { _data = v; }

        /** Evaluate the LUT */
        CUDA_DEV T eval(double y, double x) const;

        /** Evaluate the LUT from host (test function) */
        CUDA_HOST T eval_h(double y, double x);

    // Data members
    private:
        // Flags
        bool _haveData;
        bool _boundsError;
        T _refValue;
        // Coordinates
        double _xstart, _ystart, _dx, _dy;
        // LUT data
        size_t _length, _width;
        T * _data;
        // Interpolator pointer
        isce::core::dataInterpMethod _interpMethod;
        isce::cuda::core::gpuInterpolator<T> ** _interp;
        // Do I own data?
        bool _owner;

};


#endif

// end of file
