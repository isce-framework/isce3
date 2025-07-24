#include "Resample.h"

#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <isce3/core/Constants.h>

#include <isce3/cuda/core/gpuLUT2d.h>
#include <isce3/cuda/core/gpuInterpolator.h>
#include <isce3/cuda/except/Error.h>

namespace isce3::cuda::image::v2 {

using isce3::cuda::core::gpuInterpolator;
using isce3::cuda::core::gpuLUT2d;
using isce3::cuda::core::gpuSinc2dInterpolator;

using isce3::core::SINC_ONE;
using isce3::core::SINC_HALF;
using isce3::core::SINC_LEN;
using isce3::core::SINC_SUB;

__global__
void _resampleToCoordsGlobal(
    thrust::complex<float>* resampled_data_block,
    const size_t resampled_block_width,
    const size_t resampled_block_length,
    const thrust::complex<float>* input_data_block,
    const size_t input_block_width,
    const size_t input_block_length,
    thrust::complex<float>* chip,
    const double* range_input_indices,
    const double* azimuth_input_indices,
    const double startingRange,
    const double rangePixelSpacing,
    const double sensingStart,
    const double pri,                       // Pulse repetition interval, inverse of prf
    const gpuLUT2d<double> native_doppler_lut,
    gpuSinc2dInterpolator<thrust::complex<float>> interp,
    const thrust::complex<float> fill_value
)
{
    // NOTE: This function uses PRI instead of PRF since operations rely on dividing
    // by PRF. Division is more expensive than multiplication on the device, so passing
    // the reciprocal of PRF and multiplying instead is preferable.

    const auto pixel_index = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    // Prior to any data operations, return if the pixel is outside of the output grid.
    // This check is necessary because a function call from host to device must be
    // done with a multiple of the thrd_per_block pixels, but the output data size will
    // typically be smaller than this multiple. So, some calls to this function on
    // the device will be for non-existent pixels which must be discarded.
    if (pixel_index > resampled_block_width * resampled_block_length) return;

    const auto chip_size = static_cast<size_t>(SINC_ONE);

    const auto chip_pixels = chip_size * chip_size;

    const auto chip_start = pixel_index * chip_pixels;

    // The indices on the resampled data block. Assumes that range/azimuth indices
    // vectors are the same shape as the resampled data vector.
    // unit: column pixels on input array (double)
    const auto range_input_ind = range_input_indices[pixel_index];
    // unit: row pixels on input array (double)
    const auto azimuth_input_ind = azimuth_input_indices[pixel_index];

    // Skip if either the azimuth or range input index are NaN.
    if (std::isnan(azimuth_input_ind) || std::isnan(range_input_ind)) {
        resampled_data_block[pixel_index] = fill_value;
        return;
    }

    // unit: range column indices (int)
    const auto range_input_ind_int = __double2int_rd(range_input_ind);
    // unit: azimuth row indices (int)
    const auto azimuth_input_ind_int = __double2int_rd(azimuth_input_ind);
    
    // Check if chip indices could be outside radar grid minus margin to
    // account for sinc chip. Fill with fill_value and skip if chip indices
    // out of bounds.
    if (
        (range_input_ind_int < SINC_HALF) ||
        (range_input_ind_int >= (input_block_width - SINC_HALF)) ||
        (azimuth_input_ind_int < SINC_HALF) ||
        (azimuth_input_ind_int >= (input_block_length - SINC_HALF))
    ) {
        resampled_data_block[pixel_index] = fill_value;
        return;
    }

    // unit: range column indices (double)
    const auto range_input_index_remainder =
        range_input_ind - __int2double_rn(range_input_ind_int);
    // unit: azimuth row indices (double)
    const auto azimuth_input_index_remainder =
        azimuth_input_ind - __int2double_rn(azimuth_input_ind_int);

    // Slant Range at the current output pixel
    // unit: distance (meters)
    const double rg_distance = startingRange + range_input_ind * rangePixelSpacing;

    // Azimuth time at the current output pixel
    // unit: time (seconds)
    const double az_time = sensingStart + azimuth_input_ind * pri;

    // If the doppler LUT doesn't contain this coordinate, fill this pixel
    // with the given fill_value and skip it.
    if (not native_doppler_lut.contains(az_time, rg_distance)) {
        resampled_data_block[pixel_index] = fill_value;
        return;
    }
    
    // Evaluate doppler at current range and azimuth time
    // unit: frequency (radians per sample)
    const auto doppler_freq =
        native_doppler_lut.eval(az_time, rg_distance) * 2.0 * M_PI * pri;

    // Read data chip
    for (int chip_az = 0; chip_az < SINC_ONE; ++chip_az){
        // Row to read from in Azimuth coordinates
        // unit: azimuth row indices (int)
        const auto az_chip_idx = azimuth_input_ind_int + chip_az - SINC_HALF;

        // Compute doppler phase to be removed from radar data.
        // (i.e. as a unit vector on the complex plane.)
        const double doppler_phase = doppler_freq * (chip_az - SINC_HALF);
        const thrust::complex<float> doppler_phase_conj(
            std::cos(doppler_phase), -std::sin(doppler_phase));

        for (int chip_rg = 0; chip_rg < SINC_ONE; ++chip_rg) {
            // Column to read from in Range coordinates
            // unit: range column indices (int)
            const auto rg_chip_idx = range_input_ind_int + chip_rg - SINC_HALF;

            // Get the indices of this pixel on the chip block and input vectors.
            const auto chip_block_index = chip_start + chip_az * chip_size + chip_rg;
            const auto input_sample_index =
                input_block_width * az_chip_idx + rg_chip_idx;

            // Set the point at the chip indices to their value on the data
            // block, rotated by the doppler conjugate phasor.
            chip[chip_block_index] = 
                input_data_block[input_sample_index] * doppler_phase_conj;
        }
    }

    // Interpolation performed on data stripped of doppler.
    // Calculate the doppler phase shift to be reintroduced.
    const double doppler_resampled_phase =
        doppler_freq * azimuth_input_index_remainder;
    const thrust::complex<float> doppler_resampled_phasor(
        std::cos(doppler_resampled_phase),
        std::sin(doppler_resampled_phase)
    );

    // Interpolate chip
    const thrust::complex<float> interpolated_complex_val =
        interp.interpolate(
            SINC_HALF + range_input_index_remainder,
            SINC_HALF + azimuth_input_index_remainder,
            &chip[chip_start],
            chip_size,
            chip_size
        );

    // Add doppler to interpolated value
    resampled_data_block[pixel_index] = 
        interpolated_complex_val * doppler_resampled_phasor;

} // end _resampleToCoordsGlobal


/** Copy the contents of `arr` to the GPU and convert the elements to type `T`. */
template<class T, class U>
auto _copyToDeviceAs(const ConstArrayRef2D<U>& arr)
{
    thrust::device_vector<T> d_vec(arr.size());
    thrust::copy(arr.data(), arr.data() + arr.size(), d_vec.begin());
    return d_vec;
}

/** Copy the contents of `arr` to the GPU, preserving the element type. */
template<class T>
auto _copyToDevice(const ConstArrayRef2D<T>& arr)
{
    return _copyToDeviceAs<T>(arr);
}


// Interpolate tile to perform transformation
void
gpuResampleToCoords(
    ArrayRef2D<std::complex<float>> resampled_data_block,
    const ConstArrayRef2D<std::complex<float>> input_data_block,
    const ConstArrayRef2D<double> range_input_indices,
    const ConstArrayRef2D<double> azimuth_input_indices,
    const isce3::product::RadarGridParameters& radar_grid,
    const isce3::core::LUT2d<double>& native_doppler_lut,
    const std::complex<float> fill_value
) {
    // number of columns on input array
    const auto in_width = static_cast<size_t>(input_data_block.cols());
    // number of rows on input array
    const auto in_length = static_cast<size_t>(input_data_block.rows());
    // number of columns on output array
    const auto out_width = static_cast<size_t>(resampled_data_block.cols());
    // number of rows on output array
    const auto out_length = static_cast<size_t>(resampled_data_block.rows());

    const auto chip_size = SINC_ONE;

    // Number of threads per block (should always %32==0)
    const int thrd_per_block = 256;

    // Determine the number of pixels and the size of the chip array.
    const size_t num_resampled_pixels = out_width * out_length;
    const size_t num_chip_elements = num_resampled_pixels * chip_size * chip_size;

    // Instantiate the interpolator.
    // A small change over previous versions - a pointer to this object would previously
    // been passed as an argument to this function. In order to make this code
    // callable at the Python level, this has been moved here. In order to add support
    // for different interpolators, a Python binding needs to be made for these
    // interpolator objects.
    auto interp = gpuSinc2dInterpolator<thrust::complex<float>>(SINC_LEN, SINC_SUB);

    // Declare device vectors for all input data and copy the input data to them.
    auto d_input_data = _copyToDeviceAs<thrust::complex<float>>(input_data_block);
    auto d_range_indices = _copyToDevice(range_input_indices);
    auto d_azimuth_indices = _copyToDevice(azimuth_input_indices);

    gpuLUT2d<double> d_doppler(native_doppler_lut);

    // Convert std::complex to thrust::complex for invalid value.
    const thrust::complex d_fill_value(fill_value);

    // Declare the output vector.
    thrust::device_vector<thrust::complex<float>> d_resampled_data(
        num_resampled_pixels
    );

    // Declare the chip array on the device. This is a large array that will contain
    // the chip values for each pixel.
    // XXX: This seems inefficient. Look at ways to fix it.
    thrust::device_vector<thrust::complex<float>> d_chip(num_chip_elements);

    // Determine the grid of blocks needed to run this algorithm on the device.
    dim3 block(thrd_per_block);
    dim3 grid((num_resampled_pixels + (thrd_per_block - 1)) / thrd_per_block);

    // Launch the kernel on the GPU.
    _resampleToCoordsGlobal<<<grid, block>>>(
        d_resampled_data.data().get(),
        out_width,
        out_length,
        d_input_data.data().get(),
        in_width,
        in_length,
        d_chip.data().get(),
        d_range_indices.data().get(),
        d_azimuth_indices.data().get(),
        radar_grid.startingRange(),
        radar_grid.rangePixelSpacing(),
        radar_grid.sensingStart(),
        1 / radar_grid.prf(),
        d_doppler,
        interp,
        d_fill_value
    );

    // Check for any kernel errors.
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Write the output data from the device to the host.
    thrust::copy(
        d_resampled_data.begin(),
        d_resampled_data.end(),
        resampled_data_block.data()
    );
}

} // end namespace isce3::cuda::image::v2