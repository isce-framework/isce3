#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Following needed to load Swath which is needed to load SubSwaths
// CPU SubSwaths needed to load GPU SubSwaths
#include <isce3/core/Linspace.h>
#include <isce3/cuda/except/Error.h>
#include <isce3/io/IH5.h>
#include <isce3/product/RadarGridProduct.h>
#include <isce3/product/SubSwaths.h>
#include <isce3/product/Swath.h>

// What test is actually testing
#include <isce3/cuda/product/SubSwaths.h>

using CpuSubswaths = isce3::product::SubSwaths;
using OwnerGpuSubswaths = isce3::cuda::product::OwnerSubSwaths;
using ViewGpuSubswaths = isce3::cuda::product::ViewSubSwaths;


// Struct to persist various SubSwath objects.
struct gpuSubSwathsTest : public ::testing::Test {
    // CPU Swath object.
    isce3::product::Swath cpu_swath;

    // CPU SubSwath object.
    CpuSubswaths cpu_subswaths;

    // CUDA SubSwath owner and viewer objects.
    OwnerGpuSubswaths owner_gpu_subswaths;
    ViewGpuSubswaths view_gpu_subswaths;
protected:
    gpuSubSwathsTest() {
        // Following dataset only has 1 subswath
        const std::string h5path(TESTDATA_DIR "Greenland.h5");

        // Open file.
        isce3::io::IH5File ih5_file(h5path);

        // Instantiate and load a product.
        isce3::product::RadarGridProduct rdr_product(ih5_file);

        // Get swath and its subswaths.
        cpu_swath = rdr_product.swath('A');
        cpu_subswaths = cpu_swath.subSwaths();

        // Create CUDA owner and its viewer.
        owner_gpu_subswaths = OwnerGpuSubswaths(cpu_subswaths);
        view_gpu_subswaths = ViewGpuSubswaths(owner_gpu_subswaths);
    }
};

/**  Ensure CUDA and CPU SubSwaths have same number of SubSwaths and identical
  *  dimensions.
  */
TEST_F(gpuSubSwathsTest, CheckClassConstruction)
{
    EXPECT_EQ(owner_gpu_subswaths.n_subswaths(), cpu_subswaths.numSubSwaths());
    EXPECT_EQ(owner_gpu_subswaths.length(), cpu_swath.lines());
    EXPECT_EQ(owner_gpu_subswaths.width(), cpu_swath.samples());
}

/**  Kernel to check if a given az and slant range pair are in any subswaths.
  */
__global__
void containsIndex(ViewGpuSubswaths view_subswaths,
        const int* az_inds,
        const int* srg_inds,
        const size_t n_inds,
        bool* results)
{
    // Get thread index (1d grid of 1d blocks).
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid >= n_inds)
        return;

    // Call contains and save result.
    results[tid] = view_subswaths.contains(az_inds[tid], srg_inds[tid]);
}

/**  Convenience function to call kernel over multiple az and slant range
  *  index pairs.
  */
__host__
thrust::host_vector<bool> containsIndices(
        const ViewGpuSubswaths& view_subswaths,
        thrust::device_vector<int>& az_inds,
        thrust::device_vector<int>& srg_inds)
{
    const unsigned threads_per_block = 256;
    const unsigned n_blocks =
            (az_inds.size() + threads_per_block - 1) / threads_per_block;

    // Get number of inputs and create to be populated bool output of same size
    const auto n_inds = az_inds.size();
    if (srg_inds.size() != n_inds) {
        throw std::invalid_argument("az_inds.size() must be == srg_inds.size()");
    }
    thrust::device_vector<bool> dev_results(n_inds, true);

    // Launch kernel.
    containsIndex<<<n_blocks, threads_per_block>>>(
            view_subswaths,
            thrust::raw_pointer_cast(az_inds.data()),
            thrust::raw_pointer_cast(srg_inds.data()),
            n_inds,
            thrust::raw_pointer_cast(dev_results.data())
            );

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    const thrust::host_vector<bool> host_results = dev_results;
    return host_results;
}

/**  Check if start and stop indices correctly copied by comparing the contents
  *  CPU subswaths array and GPU owner subswaths' start and stop data.
  */
TEST_F(gpuSubSwathsTest, CompareValues)
{
    // Get start/stop array from one and only subswath
    const auto cpu_subswath = cpu_subswaths.getValidSamplesArray(1);

    const auto length = cpu_subswath.length();
    const size_t n_bytes_to_cp = length * sizeof(int);

    // Copy start/stop index data from GPU subswath object for comparison
    // against CPU version
    std::vector<int> vec_starts(cpu_subswath.length());
    checkCudaErrors(cudaMemcpy(&vec_starts[0],
                               owner_gpu_subswaths.ptr_to_valid_start(),
                               n_bytes_to_cp,
                               cudaMemcpyDeviceToHost));

    std::vector<int> vec_stops(cpu_subswath.length());
    checkCudaErrors(cudaMemcpy(&vec_stops[0],
                               owner_gpu_subswaths.ptr_to_valid_stop(),
                               n_bytes_to_cp,
                               cudaMemcpyDeviceToHost));

    for (int i = 0; i < length; ++i)
    {
        EXPECT_EQ(vec_starts[i], cpu_subswath(i, 0));
        EXPECT_EQ(vec_stops[i], cpu_subswath(i, 1));
    }
}

/**  Check obviously invalid az and slant range indices.
  */
TEST_F(gpuSubSwathsTest, CheckObviouslyInvalidInputs)
{
    // Invalid range index values: -1 and any >= num_samples.
    const auto num_samples = static_cast<int>(cpu_swath.lines());
    const std::vector<int> vec_srg_inds = {-1, -1, 0, num_samples,
        num_samples, num_samples + 1, num_samples + 1};
    const auto n_inds = vec_srg_inds.size();
    thrust::device_vector<int> dev_srg_inds(vec_srg_inds);

    // Index of -1 is invalid.
    const std::vector<int> vec_az_inds = {-1, 0, -1, 0, 1, 0 , 1};
    thrust::device_vector<int> dev_az_inds(vec_az_inds);

    // Check obviously invalid values.
    const auto host_test_results = containsIndices(view_gpu_subswaths,
                                                   dev_az_inds,
                                                   dev_srg_inds);

    // Check all test results are false.
    auto test_it = host_test_results.begin();
    for (; test_it != host_test_results.end(); ++test_it)
        EXPECT_EQ(*test_it, false);
}

/**  Check different az indices based per range index.
  */
TEST_F(gpuSubSwathsTest, CheckIntermediateValidRgLine)
{
    // Get index of last of subswath line.
    const auto i_last_sample = static_cast<int>(cpu_swath.lines() - 1);

    // Get first and last valid azimuth indices.
    const auto cpu_subswath = cpu_subswaths.getValidSamplesArray(1);

    // Init containers for az and srg inds to be tested, and expected results
    // to be filled in following loop.
    std::vector<int> host_srg_inds;
    std::vector<int> host_az_inds;
    std::vector<bool> expected_results;

    // Declare expected results out of loop as it doesn't change per range line.
    const std::vector<bool> tmp_results{false, false, true, true, true, false, false};

    // Create a 10 element linspace that spans samples in the subswath.
    const int n_data_sets = 10;
    auto rg_inds = isce3::core::Linspace<int>::from_interval(
            0, i_last_sample, n_data_sets);

    // Create temporary range, az, and expected value vectors for each sample
    // element in the linspace.
    for (int i = 0; i < n_data_sets; ++i)
    {
        // Bump last linspace element to last subswath line.
        const auto i_swath_line =
            i == n_data_sets - 1 ? i_last_sample: rg_inds[i];

        // Get subswath range start stop for current line.
        const auto subswath_srg_start = cpu_subswath(i_swath_line, 0);
        const auto subswath_srg_end = cpu_subswath(i_swath_line, 1);

        // Init tmp range inds and append.
        std::vector<int> tmp_srg_inds = {-1, subswath_srg_start - 1,
            subswath_srg_start, (subswath_srg_start + subswath_srg_end) / 2,
            subswath_srg_end - 1, subswath_srg_end, subswath_srg_end + 1};
        host_srg_inds.insert(host_srg_inds.end(), tmp_srg_inds.begin(),
                tmp_srg_inds.end());

        // Init tmp azimuth inds and append.
        const std::vector<int> tmp_az_inds(tmp_srg_inds.size(), i_swath_line);
        host_az_inds.insert(host_az_inds.end(), tmp_az_inds.begin(),
                tmp_az_inds.end());

        // Append expected results.
        expected_results.insert(expected_results.end(), tmp_results.begin(),
                tmp_results.end());
    }

    // Init indices on device.
    thrust::device_vector<int> dev_az_inds = host_az_inds;
    thrust::device_vector<int> dev_srg_inds = host_srg_inds;

    // Check ind values against CUDA subswath and save results.
    const auto dev_test_results = containsIndices(view_gpu_subswaths,
                                                  dev_az_inds,
                                                  dev_srg_inds);
    thrust::host_vector<bool> host_test_results = dev_test_results;

    // Compare test results againt expected results.
    auto test_it = host_test_results.begin();
    auto expected_it = expected_results.begin();
    for (;(test_it != host_test_results.end()
                and expected_it != expected_results.end());
            ++test_it, ++expected_it) {
        EXPECT_EQ(*test_it, *expected_it);
    }
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
