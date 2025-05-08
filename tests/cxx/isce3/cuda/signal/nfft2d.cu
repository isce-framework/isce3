#include <gtest/gtest.h>
#include <isce3/core/Common.h>
#include <isce3/cuda/signal/NFFT2d.h>

using isce3::cuda::signal::NFFT2d;
using isce3::cuda::signal::NFFT2dResultView;

__global__ void
interp(const NFFT2dResultView<float> result, thrust::complex<float>* z0)
{
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid > 0) {
        return;
    }

    std::array<double, 2> t = {0.0, 0.0};
    const auto z = result.interp(t, true);
    *z0 = z;
}

TEST(nfft2d, ctor)
{
    using T = float;
    using dims_t = NFFT2d<T>::dims_t;
    const dims_t m = {2, 2};
    const dims_t sizes = {201, 80};
    const dims_t fft_sizes = {512, 256};
    auto ft = NFFT2d<T>(m, sizes, fft_sizes);

    // create a spectrum equal to one everywhere
    auto npix = sizes[0] * sizes[1];
    auto spectrum = std::vector<std::complex<T>>(npix, 1.0f);

    // expect sinc in time domain, centered at [0, 0] since no phase above.
    auto result = ft.transform(sizes, {sizes[1], 1}, spectrum.data());
    auto view = NFFT2dResultView<T>(result);
    auto results_d = thrust::device_vector<thrust::complex<T>>(1);

    interp<<<1, 1>>>(view, results_d.data().get());

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::host_vector<thrust::complex<T>> results_h = results_d;
    auto z0 = results_h[0];

    EXPECT_NEAR(z0.real(), 1.0f, 1e-4);
    EXPECT_NEAR(z0.imag(), 0.0f, 1e-4);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}