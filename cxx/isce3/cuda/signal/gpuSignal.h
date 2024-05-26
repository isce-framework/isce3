#pragma once

#include "forward.h"

#include <complex>
#include <valarray>

#include <cufft.h>
#include <thrust/complex.h>

namespace isce3::cuda::signal {

template<class T>
class gpuSignal {

    public:
        // Default constructor
        gpuSignal() {};
        gpuSignal(cufftType _type);
        ~gpuSignal();

        /** \brief initiate plan for FFT in range direction
         * for a block of complex data.
         * azimuth direction is assumed to be in the direction of the
         * columns of the array.
         */
        void azimuthFFT(int ncolumns, int nrows);

        /** \brief initiate plan for FFT in azimuth direction
         * for a block of complex data.
         * range direction is assumed to be in the direction of the
         * columns of the array.
         */
        void rangeFFT(int ncolumns, int nrows);

        /** \brief initiate plan for FFT in azimuth direction
         * for a block of complex data.
         * range direction is assumed to be in the direction of the
         * columns of the array.
         */
        void FFT2D(int ncolumns, int nrows);

        /** \brief initiate cuFFT plan for a block of complex data
         *  input parameters cuFFT interface for fftw_plan_many_dft
         */
        void fftPlan(int rank, int* n, int howmany,
                    int* inembed, int istride, int idist,
                    int* onembed, int ostride, int odist);

        /** \brief next power of two*/
        void nextPowerOfTwo(size_t N, size_t& fftLength);

        /** \brief determine the required parameters for setting range FFT plans */
        inline void _configureRangeFFT(int ncolumns, int nrows);

        /** \brief determine the required parameters for setting azimuth FFT plans */
        inline void _configureAzimuthFFT(int ncolumns, int nrows);

        /** moving data in between device and host */
        void dataToDevice(std::complex<T> *input);
        void dataToDevice(std::valarray<std::complex<T>> &input);
        void dataToHost(std::complex<T> *output);
        void dataToHost(std::valarray<std::complex<T>> &output);

        /** forward transforms without intermediate return */
        void forward();

        /** forward transforms */
        void forwardC2C(std::complex<T> *input, std::complex<T> *output);
        void forwardC2C(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);
        void forwardZ2Z(std::complex<T> *input, std::complex<T> *output);
        void forwardZ2Z(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);
        void forwardD2Z(T *input, std::complex<T> *output);

        void forward(std::complex<T> *input, std::complex<T> *output);
        void forward(std::valarray<std::complex<T>> &input,
                     std::valarray<std::complex<T>> &output);

        void forwardDevMem(thrust::complex<float> *input,
                           thrust::complex<float> *output);

        void forwardDevMem(thrust::complex<double> *input,
                           thrust::complex<double> *output);

        void forwardDevMem(thrust::complex<T> *dataInPlace);

        /** inverse transforms using existing device memory **/
        void inverse();

        /** inverse transforms */
        void inverseC2C(std::complex<T> *input, std::complex<T> *output);
        void inverseC2C(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);
        void inverseZ2Z(std::complex<T> *input, std::complex<T> *output);
        void inverseZ2Z(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);
        void inverseZ2D(std::complex<T> *input, T *output);

        void inverse(std::complex<T> *input, std::complex<T> *output);
        void inverse(std::valarray<std::complex<T>> &input,
                     std::valarray<std::complex<T>> &output);

        void inverseDevMem(thrust::complex<float> *input,
                           thrust::complex<float> *output);

        void inverseDevMem(thrust::complex<double> *input,
                           thrust::complex<double> *output);

        void inverseDevMem(thrust::complex<T> *dataInPlace);

        /** upsample **/
        void upsample(std::valarray<std::complex<T>> &input,
                      std::valarray<std::complex<T>> &output,
                      int row,
                      int ncols,
                      int upsampleFactor);
        void upsample(std::valarray<std::complex<T>> &input,
                      std::valarray<std::complex<T>> &output,
                      int row,
                      int ncols,
                      int upsampleFactor,
                      std::valarray<std::complex<T>> &shiftImpact);

        int getRows() {return _rows;};
        int getColumns() {return _columns;};
        int getNumElements() {return _n_elements;};

        thrust::complex<T>* getDevicePtr() {return _d_data;};

    private:
        cufftHandle _plan;
        bool _plan_set;
        cufftType _cufft_type;

        // FFT plan parameters
        int _rank;
        int _n[2];
        int _howmany;
        int _inembed[2];
        int _istride;
        int _idist;
        int _onembed[2];
        int _ostride;
        int _odist;
        int _n_elements;
        int _rows;
        int _columns;

        // device memory pointers
        thrust::complex<T> *_d_data;
        bool _d_data_set;
};

template<class T>
void upsample(gpuSignal<T> &fwd,
        gpuSignal<T> &inv,
        thrust::complex<T> *input,
        thrust::complex<T> *output);

template<class T>
void upsample(gpuSignal<T> &fwd,
        gpuSignal<T> &inv,
        thrust::complex<T> *input,
        thrust::complex<T> *output,
        thrust::complex<T> *shiftImpact);

template<class T>
void upsample(gpuSignal<T> &fwd,
        gpuSignal<T> &inv,
        std::valarray<std::complex<T>> &input,
        std::valarray<std::complex<T>> &output);

template<class T>
void upsample(gpuSignal<T> &fwd,
        gpuSignal<T> &inv,
        std::valarray<std::complex<T>> &input,
        std::valarray<std::complex<T>> &output,
        std::valarray<std::complex<T>> &shiftImpact);

} // namespace isce3::cuda::signal
