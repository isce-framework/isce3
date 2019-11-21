#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <isce/cuda/except/Error.h>
#include <cufft.h>
#include <thrust/device_vector.h>

// Just ensure that checkCudaErrors throws an ISCE CudaError
TEST(CheckCudaErrors, BasicThrow) {
    EXPECT_THROW(checkCudaErrors(cudaSetDevice(-1)),
                 isce::cuda::except::CudaError<cudaError_t>);
}

// Check that a non-zero cufftResult throws an error
// Make sure an error is thrown when executing an FFT before setting up a plan
TEST(CheckCudaErrors, CufftResult) {
    cufftHandle plan;
    thrust::device_vector<cufftComplex> signal;
    cufftComplex* data = signal.data().get();
    EXPECT_THROW(checkCudaErrors(cufftExecC2C(plan, data, data, CUFFT_FORWARD)),
                 isce::cuda::except::CudaError<cufftResult>);
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
