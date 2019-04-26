#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <isce/cuda/except/Error.h>

// Just ensure that checkCudaErrors throws an ISCE CudaError
TEST(CheckCudaErrors, BasicThrow) {
    EXPECT_THROW(checkCudaErrors(cudaSetDevice(-1)),
                 isce::cuda::except::CudaError<cudaError_t>);
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
