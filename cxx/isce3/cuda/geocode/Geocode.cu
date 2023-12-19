#include <cmath>
#include <cuda_runtime.h>
#include <gdal_priv.h>
#include <limits>
#include <thrust/complex.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>

#include <pyre/journal.h>

#include <isce3/core/Ellipsoid.h>
#include <isce3/core/EMatrix.h>
#include <isce3/core/Projections.h>
#include <isce3/cuda/container/RadarGeometry.h>
#include <isce3/cuda/core/OrbitView.h>
#include <isce3/cuda/core/gpuProjections.h>
#include <isce3/cuda/except/Error.h>
#include <isce3/cuda/geometry/gpuDEMInterpolator.h>
#include <isce3/cuda/geometry/gpuGeometry.h>
#include <isce3/except/Error.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/loadDem.h>
#include <isce3/product/GeoGridParameters.h>

#include "Geocode.h"
#include "MaskedMinMax.h"

using isce3::core::Vec3;
using isce3::io::Raster;

using isce3::core::SINC_HALF;
using isce3::core::SINC_ONE;
using isce3::cuda::core::InterpolatorHandle;
using DeviceOrbitView = isce3::cuda::core::OrbitView;
using isce3::cuda::core::ProjectionBaseHandle;
using isce3::geometry::detail::Geo2RdrParams;

template<typename T>
using DeviceInterp = isce3::cuda::core::gpuInterpolator<T>;

template<typename T>
using DeviceLUT2d = isce3::cuda::core::gpuLUT2d<T>;

using GpuOwnerSubSwaths = isce3::cuda::product::OwnerSubSwaths;
using GpuViewSubSwaths = isce3::cuda::product::ViewSubSwaths;

namespace isce3::cuda::geocode {


/**  Coverts a batch of rows from input geogrid into radar coordinates. Outputs
 *  the pixel-space coordinates (x,y) of each resulting (range, azimuth) pair
 *  with respect to specified radargrid, as well as mask invalid pixels (out
 *  of bounds, failed to converge in geo2rdr, or not contained in
 *  azTimeCorrections, sRangeCorrections, or nativeDoppler).
 *
 * \param[out] rdr_x            pointer to device_vector of computed radar grid
 *                              x / slant range indices
 * \param[out] rdr_y            pointer to device_vector of computed radar grid
 *                              y / azimuth time indices
 * \param[out] mask             pointer to device_vector mask of valid pixels
 *                              Mask follows numpy mask_array convention where
 *                              True is masked.
 * \param[in] ellipsoid         Ellipsoid based on output geogrid coordinate
 *                              system
 * \param[in] orbit             Orbit associated with radar data
 * \param[in] dem               DEM interpolator. Maybe be of different
 *                              coordinate system than output geogrid.
 * \param[in] nativeDoppler     doppler centroid of data associated with radar
 *                              grid, in Hz, as a function of azimuth and range
 * \param[in] imageDoppler      image grid doppler
 * \param[in] wvl               wavelength
 * \param[in] side              look side
 * \param[in] geo2rdr_params    geo2rdr params
 * \param[in] geogrid           Geogrid defining output product/rasters
 * \param[in] radargrid         Radar grid decribing rasters to be geocoded
 * \param[in] line_start        Starting line of block
 * \param[in] block_size        Number of elements in a block
 * \param[in] proj              Projection used to covert geogrid XYZ to LLH
 *                              of output coordinate system
 * \param[in] azTimeCorrection  geo2rdr azimuth additive correction, in
 *                              seconds, as a function of azimuth and range
 * \param[in] sRangeCorrection  geo2rdr slant range additive correction, in
 *                              seconds, as a function of azimuth and range
 * \param[in]  subswaths        view of subswath mask representing valid
 *                              portions of a swath
 */
__global__ void geoToRdrIndices(double* rdr_x, double* rdr_y, bool* mask,
        const isce3::core::Ellipsoid ellipsoid, const DeviceOrbitView orbit,
        isce3::cuda::geometry::gpuDEMInterpolator dem,
        const DeviceLUT2d<double> nativeDoppler,
        const DeviceLUT2d<double> imageDoppler,
        const double wvl,
        const isce3::core::LookSide side, const Geo2RdrParams geo2rdr_params,
        const isce3::product::GeoGridParameters geogrid,
        const RadarGridParamsLite radargrid, const size_t line_start,
        const size_t block_size, isce3::cuda::core::ProjectionBase** proj,
        const DeviceLUT2d<double> azTimeCorrection,
        const DeviceLUT2d<double> sRangeCorrection,
        const GpuViewSubSwaths subswaths)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid >= block_size)
        return;

    const size_t block_line = tid / geogrid.width();
    const size_t pixel = tid % geogrid.width();
    const size_t line = line_start + block_line;

    // x and y coordinates of the output geocoded grid
    const Vec3 xyz {geogrid.startX() + geogrid.spacingX() * (0.5 + pixel),
            geogrid.startY() + geogrid.spacingY() * (0.5 + line), 0.0};

    Vec3 llh;
    (*proj)->inverse(xyz, llh);
    llh[2] = dem.interpolateLonLat(llh[0], llh[1]);

    // slant range to be computed by geo2rdr
    double srange;
    // initial guess for azimuth time to be updated by geo2rdr
    double aztime = radargrid.sensing_mid;

    // returns 0 if geo2rdr converges else 1
    int converged = isce3::cuda::geometry::geo2rdr(llh, ellipsoid, orbit,
            imageDoppler, &aztime, &srange, wvl, side,
            geo2rdr_params.threshold, geo2rdr_params.maxiter,
            geo2rdr_params.delta_range);

    // Bool to track if aztime and slant range is not contained in any of the
    // following LUT2d's: azTimeCorrection, sRangeCorrection, and nativeDoppler.
    // When set to true, this geogrid point will be marked as invalid
    bool mask_this_pixel = false;

    // Apply timing corrections
    if (azTimeCorrection.contains(aztime, srange)) {
        const auto aztimeCor = azTimeCorrection.eval(aztime, srange);
        aztime += aztimeCor;
    } else
        mask_this_pixel = true;

    if (sRangeCorrection.contains(aztime, srange)) {
        const auto srangeCor = sRangeCorrection.eval(aztime, srange);
        srange += srangeCor;
    } else
        mask_this_pixel = true;

    if (!nativeDoppler.contains(aztime, srange))
        mask_this_pixel = true;

    // convert aztime and range to indices as double, int, and size_t
    // ensure size_t values do convert negative value
    const double y = (aztime - radargrid.sensing_start) * radargrid.prf;
    const auto y_int = static_cast<int>(std::floor(y));
    const auto y_size_t = y_int < 0 ? 0 : static_cast<size_t>(y_int);
    const double x = (srange - radargrid.starting_range) / radargrid.range_pixel_spacing;
    const auto x_int = static_cast<int>(std::floor(x));
    const auto x_size_t = x_int < 0 ? 0 : static_cast<size_t>(x_int);

    if (!subswaths.contains(y_int, x_int))
        mask_this_pixel = true;

    // check if indices in bounds of radar grid
    const bool not_in_rdr_grid =
            y_int < 0 || y_size_t >= radargrid.length ||
            x_int < 0 || x_size_t >= radargrid.width;

    const bool invalid_index = not_in_rdr_grid || converged == 0 || mask_this_pixel;
    rdr_y[tid] = invalid_index ? 0.0 : y;
    rdr_x[tid] = invalid_index ? 0.0 : x;
    mask[tid] = invalid_index;
}

/** Interpolate radar block data to block indices calculated in geoToRdrIndices
 *
 * \param[out] geo_data_block   pointer to device vector of geocoded data of
 *                              current block
 * \param[in] rdr_x             pointer to device vector of radar grid x / az
 *                              time indices of current block
 * \param[in] rdr_y             pointer to device vector of radar grid y / range
 *                              indices of current block
 * \param[in] mask              pointer to device vector of a mask / valid
 *                              pixels of current block
 * \param[in] rdr_data_block    pointer to device vector of radar data of
 current block
 * \param[in] width             width of rdr_data_block
 * \param[in] length            length of rdr_data_block
 * \param[in] block_size        number of elements in a block
 * \param[in] az_1st_line       offset applied to az time indices to correctly
 *                              access current block
 * \param[in] range_1st_pixel   offset applied to slant range indices to
 *                              correctly access current block
 * \param[in] invalid_value     value assigned to invalid geogrid pixels
 * \param[in] interp            interpolator used to interpolate radar data to
                                specified geogrid
 * \param[in] is_sinc_interp    True if sinc interpolation is to be used.
                                Allows the kernel to call sinc interpolator
                                differently from other interpolators.
 * \param[in,out] sinc_chips    Array for chips of all points to be geocoded.
                                Not ideal, nor efficient. Done to match
                                sinc interpolation in CUDA resample SLC.
 */
template<class T>
__global__ void interpolate(T* geo_data_block,
        const double* rdr_x,
        const double* rdr_y,
        const bool* mask,
        const T* rdr_data_block,
        const size_t width,
        const size_t length,
        const size_t block_size,
        const double az_1st_line,
        const double range_1st_pixel,
        const T invalid_value,
        DeviceInterp<T>** interp,
        const bool is_sinc_interp,
        T* sinc_chips)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid >= block_size)
        return;

    // Determine radar indices with respect to current block. rdr_y/x are
    // indices for the entire raster thus an offset of block first az/y or
    // range/x has to be accounted for.
    auto rdry = rdr_y[tid] - az_1st_line;
    auto rdrx = rdr_x[tid] - range_1st_pixel;

    // Check if indices are in bounds of radar block
    // Add margin to account for possibly deficient interpolator boundary checks
    constexpr double extra_margin = 4.0; // magic number for SINC_HALF
    bool out_of_bounds = rdrx < extra_margin || rdry < extra_margin ||
                         rdrx >= width - extra_margin ||
                         rdry >= length - extra_margin;

    // Default to invalid value. Interpolate only if in bounds and not masked.
    T interp_val = invalid_value;
    if (!(out_of_bounds || mask[tid])) {
        // Interpolate differently if sinc interpolation because of chip
        // construction. 'if/else' does not present a divergence issue as all
        // the kernels per block share the same interpolation method.
        if (is_sinc_interp) {
            // Get integer rdr x/y in order to access array
            const int rdr_x_int = __double2int_rd(rdrx);
            const int rdr_y_int = __double2int_rd(rdry);

            // Get float factional of rdr x/y to compute coordinates of sinc
            // interpolation
            const double rdr_x_frac = rdrx - __int2double_rn(rdr_x_int);
            const double rdr_y_frac = rdry - __int2double_rn(rdr_y_int);

            // Index for start of current chip. SINC_ONE**2 stride accounts
            // for all chips for all SINC_ONE**2 elements of each chip.
            const auto chip_start = tid * SINC_ONE * SINC_ONE;

            // Load data chip from radar data block
            for (int i_row = 0; i_row < SINC_ONE; ++i_row) {

                // Row in radar data to read from
                const int rdr_blk_row = rdr_y_int + i_row - SINC_HALF;

                for (int i_col = 0; i_col < SINC_ONE; ++i_col) {

                    // Column in radar data to read from
                    const int rdr_blk_col = rdr_x_int + i_col - SINC_HALF;

                    // Chip index w.r.t chip_start
                    sinc_chips[chip_start + i_row * SINC_ONE + i_col] =
                            rdr_data_block[rdr_blk_row * width + rdr_blk_col];
                }
            }

            // Perform sinc interpolation on chip
            interp_val = (*interp)->interpolate(
                    SINC_HALF + rdr_x_frac, SINC_HALF + rdr_y_frac,
                    &sinc_chips[chip_start], SINC_ONE, SINC_ONE);
        }
        else {
            interp_val = (*interp)->interpolate(
                    rdrx, rdry, rdr_data_block, width, length);
        }
    }

    geo_data_block[tid] = interp_val;
}

/* Check if int dtype only used with nearest neighbor interpolation
 *
 * \param[in] dtype                 Data type to be interpolated
 * \param[in] data_interp_method    Method of interpolation to be used
 */
void rasterDtypeInterpCheck(const int dtype,
        const isce3::core::dataInterpMethod& data_interp_method)
{
    if ((dtype == GDT_Byte || dtype == GDT_UInt32) &&
            data_interp_method != isce3::core::NEAREST_METHOD) {
        std::string err_str {
                "int type of raster can only use nearest neighbor interp"};
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), err_str);
    }
}

void invalidValueCheck(const int dtype, const double invalid_value)
{
    const auto is_int_type = dtype == GDT_Byte or dtype == GDT_UInt32;

    if (isnan(invalid_value) and is_int_type) {
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "int-type datatype can not have NaN invalid value");
    }

    // Check if invalid value is in bounds for unsigned int
    if ((dtype == GDT_UInt32)
            and (invalid_value > std::numeric_limits<unsigned int>::max())) {
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "unsigned int datatype invalid value exceeds type max");
    }

    // Check if invalid value is in bounds for unsigned char
    if ((dtype == GDT_Byte)
            and (invalid_value > std::numeric_limits<unsigned char>::max())) {
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "unsigned char datatype invalid value exceeds type max");
    }

    // Check if invalid value is in bounds for float32 types
    if ((dtype == GDT_Float32 or dtype == GDT_CFloat32)
            and (abs(invalid_value) > std::numeric_limits<float>::max())) {
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "float32-type datatype invalid value exceeds type max");
    }
}


__host__ Geocode::Geocode(const isce3::product::GeoGridParameters& geogrid,
        const isce3::container::RadarGeometry& rdr_geom,
        const size_t lines_per_block) :
    _n_blocks((geogrid.length() + lines_per_block - 1) / lines_per_block),
    _geogrid(geogrid),
    _rdr_geom(rdr_geom),
    _ellipsoid(isce3::core::makeProjection(_geogrid.epsg())->ellipsoid()),
    // first line/pixel members below default to radar grid extent to ensure
    // first min() correctly
    // last line/pixel members below default to 0 to ensure first max()
    // correctly
    _az_first_line(_rdr_geom.radarGrid().length() - 1),
    _az_last_line(0),
    _range_first_pixel(rdr_geom.radarGrid().width() - 1),
    _range_last_pixel(0),
    _geo_block_length(lines_per_block),
    _proj_handle(geogrid.epsg())
{
    // init light weight radar grid
    _radar_grid.sensing_start = rdr_geom.radarGrid().sensingStart();
    _radar_grid.sensing_mid = rdr_geom.radarGrid().sensingMid();
    _radar_grid.prf = rdr_geom.radarGrid().prf();
    _radar_grid.starting_range = rdr_geom.radarGrid().startingRange();
    _radar_grid.range_pixel_spacing = rdr_geom.radarGrid().rangePixelSpacing();
    _radar_grid.length = rdr_geom.gridLength();
    _radar_grid.width = rdr_geom.gridWidth();

    // Determine max number of elements per block. Last block to be processed
    // may not contain the max number of elements.
    auto n_elem = _geo_block_length * _geogrid.width();

    // Resize all device vectors to max block size.
    _radar_x.resize(n_elem);
    _radar_y.resize(n_elem);
    _mask.resize(n_elem);
}

void Geocode::setBlockRdrCoordGrid(const size_t block_number,
        Raster& dem_raster,
        const isce3::core::dataInterpMethod dem_interp_method,
        const isce3::cuda::container::RadarGeometry& dev_rdr_geom,
        const isce3::geometry::detail::Geo2RdrParams geo2rdr_params,
        const DeviceLUT2d<double>& nativeDoppler,
        const DeviceLUT2d<double>& azTimeCorrection,
        const DeviceLUT2d<double>& sRangeCorrection,
        const GpuViewSubSwaths& view_subswaths)
{
    // make sure block index does not exceed actual number of blocks
    if (block_number >= _n_blocks) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "block number exceeds max number of blocks");
    }

    // Get block extents (of the geocoded grid)
    _line_start = block_number * _geo_block_length;

    // Set block sizes for everything but last block
    size_t block_size = _geo_block_length * _geogrid.width();

    // Adjust for last block sized assuming it is sized differently than others
    if (block_number == (_n_blocks - 1)) {
        _geo_block_length = _geogrid.length() - _line_start;
        block_size = _geo_block_length * _geogrid.width();
    }

    // Resize blocks accordingly
    _radar_x.resize(block_size);
    _radar_y.resize(block_size);
    _mask.resize(block_size);

    // prepare device DEMInterpolator
    int dem_margin_in_pixels = 50;
    isce3::geometry::DEMInterpolator host_dem_interp =
            isce3::geometry::DEMRasterToInterpolator(dem_raster, _geogrid,
                    _line_start, _geo_block_length, _geogrid.width(),
                    dem_margin_in_pixels, dem_interp_method);
    isce3::cuda::geometry::gpuDEMInterpolator dev_dem_interp(host_dem_interp);

    // Create geogrid on device
    {
        const unsigned threads_per_block = 256;
        const unsigned n_blocks =
                (block_size + threads_per_block - 1) / threads_per_block;

        geoToRdrIndices<<<n_blocks, threads_per_block>>>(_radar_x.data().get(),
                _radar_y.data().get(), _mask.data().get(), _ellipsoid,
                dev_rdr_geom.orbit(), dev_dem_interp,
                nativeDoppler, dev_rdr_geom.doppler(),
                dev_rdr_geom.wavelength(), dev_rdr_geom.lookSide(),
                geo2rdr_params, _geogrid, _radar_grid, _line_start, block_size,
                _proj_handle.get_proj(), azTimeCorrection, sRangeCorrection,
                view_subswaths);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // Mask is utilized to _radar_x and _radar_y when determining respective
    // min and max values which are then used to determine corresponding radar
    // data to read from radar raster
    // Find index of unmasked min and max values in x/range
    const auto [rdr_x_min, rdr_x_max] = masked_minmax(_radar_x, _mask);

    _range_first_pixel = std::min(_rdr_geom.radarGrid().width() - 1,
            static_cast<size_t>(std::floor(rdr_x_min)));

    _range_last_pixel = std::max(static_cast<size_t>(0),
            static_cast<size_t>(std::ceil(rdr_x_max) - 1));

    // Find index of unmasked min and max values in y/azimuth
    const auto [rdr_y_min, rdr_y_max] = masked_minmax(_radar_y, _mask);

    _az_first_line = std::min(_rdr_geom.radarGrid().length() - 1,
            static_cast<size_t>(std::floor(rdr_y_min)));

    _az_last_line = std::max(static_cast<size_t>(0),
            static_cast<size_t>(std::ceil(rdr_y_max) - 1));

    // Extra margin for interpolation to avoid gaps between blocks in output
    const size_t interp_margin {5};

    // Not using std::max to account for possibility of wraparound resulting
    // subtracting a larger size_t from a smaller size_t
    _range_first_pixel =
        interp_margin > _range_first_pixel
        ? 0
        : _range_first_pixel - interp_margin;
    _range_last_pixel = std::min(_rdr_geom.radarGrid().width() - 1,
                                   _range_last_pixel + interp_margin);

    _az_first_line =
        interp_margin > _az_first_line ? 0 : _az_first_line - interp_margin;
    _az_last_line = std::min(_rdr_geom.radarGrid().length() - 1,
                             _az_last_line + interp_margin);

    // check if block entirely masked
    bool all_masked = std::isnan(rdr_y_min);

    // if mask not entirely masked, then set non zero dimensions
    _rdr_block_length = all_masked ? 0 : _az_last_line - _az_first_line + 1;
    _rdr_block_width =
            all_masked ? 0 : _range_last_pixel - _range_first_pixel + 1;

    pyre::journal::debug_t debug(
            "isce.cuda.geocode.Geocode.setBlockRdrCoordGrid");
    if (all_masked) {
        debug << block_number
              << " is out of bounds. calls geocodeRasterBlock will \
            not geocode."
              << pyre::journal::endl;
    }
}

void Geocode::ensureRasterConsistency(
        const std::vector<std::reference_wrapper<isce3::io::Raster>>& output_rasters,
        const std::vector<std::reference_wrapper<isce3::io::Raster>>& input_rasters,
        const std::vector<isce3::core::dataInterpMethod>& interp_methods,
        const std::vector<GDALDataType>& raster_datatypes,
        const std::vector<double>& invalid_values_double) const
{
    // get number of expected of raster pairs and corresponding interpolation
    // methods, raster datatypes, and invalid_values_double from number
    // of output_rasters
    const auto n_raster_pairs = output_rasters.size();

    // check if input and output vectors are of same length
    if (n_raster_pairs != input_rasters.size()) {
        throw isce3::except::LengthError(
                ISCE_SRCINFO(), "number of input and output rasters not equal");
    }

    // Check if interp_methods, raster_datatypes, and invalid_values_double all have
    // the same number of elements
    if (n_raster_pairs != interp_methods.size()) {
        throw isce3::except::LengthError(
                ISCE_SRCINFO(), "number of interp methods and output rasters not equal");
    }
    if (n_raster_pairs != raster_datatypes.size())
        throw isce3::except::LengthError(
                ISCE_SRCINFO(), "number of interpolation methods and data types not equal");

    if (n_raster_pairs != invalid_values_double.size())
        throw isce3::except::LengthError(
                ISCE_SRCINFO(), "number of interpolation methods and invalid values not equal");

    // For each raster input/output pair, ensure datatypes are consistent with
    // expected types
    for (size_t i_raster = 0; i_raster < n_raster_pairs; ++i_raster)
    {
        const int in_dtype = input_rasters[i_raster].get().dtype();
        const int out_dtype = output_rasters[i_raster].get().dtype();
        const int expected_dtype = raster_datatypes[i_raster];

        if (in_dtype != out_dtype) {
            throw isce3::except::InvalidArgument(
                    ISCE_SRCINFO(), "input and output rasters not same datatype");
        }

        if (in_dtype != expected_dtype) {
            throw isce3::except::InvalidArgument(
                    ISCE_SRCINFO(), "input and output rasters datatype does not match expected datatype");
        }

        // Get interpolation method to set interpolator within handler
        const auto interp_method = interp_methods[i_raster];

        // Check if raster types consistent with data interp method
        rasterDtypeInterpCheck(expected_dtype, interp_method);

        // Check if raster invalid types are in type limits or can be assigned NaN
        const double invalid_value_double = invalid_values_double[i_raster];
        invalidValueCheck(expected_dtype, invalid_value_double);
    }
}

template<class T>
void Geocode::geocodeRasterBlock(Raster& output_raster, Raster& input_raster,
        const std::shared_ptr<
            isce3::cuda::core::InterpolatorHandleVirtual>& interp_handle_ptr,
        const std::any& invalid_value_any, const bool is_sinc_interp)
{
    // determine number of elements in output vector
    const auto n_elem_out = _geo_block_length * _geogrid.width();

    // Get raw pointer from shared_ptr and recast pointer type
    const auto raw_interp_handle_ptr =
        dynamic_cast<InterpolatorHandle<T>*>(interp_handle_ptr.get());
        if (!raw_interp_handle_ptr) {
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "interp_handle_ptr must be of type InterpolatorHandle<T>*");
        }
    DeviceInterp<T>** interp = raw_interp_handle_ptr->getInterp();

    const T block_invalid_value = std::any_cast<T>(invalid_value_any);

    // 0 width indicates current block is out of bounds
    if (_rdr_block_width == 0) {
        // set entire block to invalid value
        thrust::host_vector<T> h_geo_data_block =
                thrust::host_vector<T>(n_elem_out, block_invalid_value);
        output_raster.setBlock(&h_geo_data_block[0], 0, _line_start,
                _geogrid.width(), _geo_block_length, 1);

        pyre::journal::debug_t debug(
                "isce.cuda.geocode.Geocode.geocodeRasterBlock");
        debug << "Unable to geocode raster due to block being out of bounds."
              << pyre::journal::endl;
        return;
    }

    // load raster block on host
    const auto n_elem_in = _rdr_block_length * _rdr_block_width;
    thrust::host_vector<T> h_rdr_data_block(n_elem_in);
    input_raster.getBlock(&h_rdr_data_block[0], _range_first_pixel,
            _az_first_line, _rdr_block_width, _rdr_block_length, 1);

    // copy input raster block to device
    thrust::device_vector<T> d_rdr_data_block = h_rdr_data_block;

    // prepare output geocode raster block
    thrust::device_vector<T> d_geo_data_block(n_elem_out, block_invalid_value);

    // Perform interpolation on device
    {
        // Allocate chips for sinc interpolation if needed. Each point to be
        // sinc interpolated has it chip allocated in advance. This is VERY
        // inefficient use of memory and soly done to minimize churn and meet
        // NISAR delivery deadline. A far more efficient method of
        // sinc interpolation would be to radar block data to interpolator
        // and have the interpolator access chip members there. This would
        // require a refactor to CUDA resample which is out of scope of the
        // change to introduce sinc interpolation to CUDA geocode.
        thrust::device_vector<T> sinc_chips;
        if (is_sinc_interp)
            sinc_chips.resize(n_elem_out * SINC_ONE * SINC_ONE);

        const unsigned threads_per_block = 256;
        const unsigned n_blocks =
                (n_elem_out + threads_per_block - 1) / threads_per_block;
        interpolate<<<n_blocks, threads_per_block>>>(
                d_geo_data_block.data().get(), _radar_x.data().get(),
                _radar_y.data().get(), _mask.data().get(),
                d_rdr_data_block.data().get(), _rdr_block_width,
                _rdr_block_length, n_elem_out,
                static_cast<double>(_az_first_line),
                static_cast<double>(_range_first_pixel), block_invalid_value,
                interp, is_sinc_interp,
                thrust::raw_pointer_cast(sinc_chips.data()));

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // set output raster block from device
    thrust::host_vector<T> h_geo_data_block = d_geo_data_block;

    output_raster.setBlock(&h_geo_data_block[0], 0, _line_start,
            _geogrid.width(), _geo_block_length, 1);
}

void Geocode::geocodeRasters(
        std::vector<std::reference_wrapper<isce3::io::Raster>>& output_rasters,
        std::vector<std::reference_wrapper<isce3::io::Raster>>& input_rasters,
        const std::vector<isce3::core::dataInterpMethod>& interp_methods,
        const std::vector<GDALDataType>& raster_datatypes,
        const std::vector<double>& invalid_values_double,
        Raster& dem_raster,
        const isce3::core::LUT2d<double>& hostNativeDoppler,
        const isce3::core::LUT2d<double>& hostAzTimeCorrection,
        const isce3::core::LUT2d<double>& hostSRangeCorrection,
        const isce3::product::SubSwaths* subswaths,
        const isce3::core::dataInterpMethod dem_interp_method,
        const double threshold, const int maxiter, const double dr)
{
    ensureRasterConsistency(output_rasters, input_rasters, interp_methods,
            raster_datatypes, invalid_values_double);

    const auto n_raster_pairs = output_rasters.size();

    // Interpolation handles for each of the different rasters
    // Stored as shared pointers to allow to storage of different
    // types of templated derived class
    std::vector<std::shared_ptr<isce3::cuda::core::InterpolatorHandleVirtual>> _data_interp_handles;

    // Invalid values for each of the different rasters
    std::vector<std::any> _invalid_values;

    // geo2rdr params used in radar index calculation
    isce3::geometry::detail::Geo2RdrParams geo2rdr_params{threshold, maxiter, dr};

    // 2D LUT Doppler of the SLC image
    auto nativeDoppler = DeviceLUT2d<double>(hostNativeDoppler);

    // geo2rdr azimuth additive correction, in seconds, as a function of
    // azimuth and range
    auto azTimeCorrection = DeviceLUT2d<double>(hostAzTimeCorrection);;

    // geo2rdr slant range additive correction, in seconds, as a function of
    // azimuth and range
    auto sRangeCorrection = DeviceLUT2d<double>(hostSRangeCorrection);

    // Prepare interpolator handler and invalid value for each raster
    for (size_t i_raster = 0; i_raster < n_raster_pairs; ++i_raster) {
        // Get data type of current expected raster to determine interpolator
        // handler and invalid value typea for current expected raster
        const int dtype = raster_datatypes[i_raster];

        // Get interpolation method to set interpolator within handler
        const auto interp_method = interp_methods[i_raster];

        // Get invalid value to be converted (maybe)
        const double invalid_value = invalid_values_double[i_raster];

        // Determine invalid values per type
        float invalid_float;
        double invalid_double;
        unsigned char invalid_unsigned_char;
        unsigned int invalid_unsigned_int;
        if (std::isnan(invalid_value)) {
            invalid_float = std::numeric_limits<float>::quiet_NaN();
            invalid_double = std::numeric_limits<double>::quiet_NaN();
        } else {
            invalid_float = static_cast<float>(invalid_value);
            invalid_double = invalid_value;
            invalid_unsigned_char = static_cast<unsigned char>(invalid_value);
            invalid_unsigned_int = static_cast<unsigned int>(invalid_value);
        }

        // Assign correct interpolator handle and invalid values to respective
        // class member containers by type
        switch (dtype) {
        case GDT_Float32: {
            _data_interp_handles.push_back(std::make_shared<
                    InterpolatorHandle<float>>(interp_method));
            _invalid_values.push_back(invalid_float);
            break;
        }
        case GDT_CFloat32: {
            _data_interp_handles.push_back(std::make_shared<
                    InterpolatorHandle<thrust::complex<float>>>(interp_method));
            _invalid_values.push_back(thrust::complex<float>(
                        invalid_float, invalid_float));
            break;
        }
        case GDT_Float64: {
            _data_interp_handles.push_back(std::make_shared<
                    InterpolatorHandle<double>>(interp_method));
            _invalid_values.push_back(invalid_double);
            break;
        }
        case GDT_CFloat64: {
            _data_interp_handles.push_back(std::make_shared<
                    InterpolatorHandle<thrust::complex<double>>>(interp_method));
            _invalid_values.push_back(thrust::complex<double>(
                        invalid_double, invalid_double));
            break;
        }
        case GDT_Byte: {
            _data_interp_handles.push_back(std::make_shared<
                    InterpolatorHandle<unsigned char>>(interp_method));
            _invalid_values.push_back(invalid_unsigned_char);
            break;
        }
        case GDT_UInt32: {
            _data_interp_handles.push_back(std::make_shared<
                    InterpolatorHandle<unsigned int>>(interp_method));
            _invalid_values.push_back(invalid_unsigned_int);
            break;
        }
        default: {
            throw isce3::except::RuntimeError(
                    ISCE_SRCINFO(), "unsupported datatype");
        }
        }
    }

    // copy RadarGeometry to device
    const auto dev_rdr_geom = isce3::cuda::container::RadarGeometry(_rdr_geom);

    // copy Subswaths to device as GpuOwnerSubSwaths object or init default
    auto dev_owner_subswaths = subswaths ?
        GpuOwnerSubSwaths(*subswaths) : GpuOwnerSubSwaths();

    // create GpuViewSubSwaths object with pointers that access
    // GpuOwnerSubSwaths object
    const auto dev_view_subswaths = GpuViewSubSwaths(dev_owner_subswaths);

    // iterate over blocks
    for (size_t i_block = 0; i_block < _n_blocks; ++i_block) {

        // compute then set radar coords for each geocode obj for curret block
        setBlockRdrCoordGrid(i_block,
                             dem_raster,
                             dem_interp_method,
                             dev_rdr_geom,
                             geo2rdr_params,
                             nativeDoppler,
                             azTimeCorrection,
                             sRangeCorrection,
                             dev_view_subswaths);

        // iterate over rasters and geocode with associated datatype and
        // interpolation method
        for (size_t i_raster = 0; i_raster < n_raster_pairs; ++i_raster) {
            const int dtype = input_rasters[i_raster].get().dtype();
            const auto interp_method = interp_methods[i_raster];
            const bool is_sinc_interp = interp_method == isce3::core::SINC_METHOD;
            switch (dtype) {
            case GDT_Float32: {
                geocodeRasterBlock<float>(
                        output_rasters[i_raster], input_rasters[i_raster],
                        _data_interp_handles[i_raster],
                        _invalid_values[i_raster],
                        is_sinc_interp);
                break;
            }
            case GDT_CFloat32: {
                geocodeRasterBlock<thrust::complex<float>>(
                        output_rasters[i_raster], input_rasters[i_raster],
                        _data_interp_handles[i_raster],
                        _invalid_values[i_raster],
                        is_sinc_interp);
                break;
            }
            case GDT_Float64: {
                geocodeRasterBlock<double>(
                        output_rasters[i_raster], input_rasters[i_raster],
                        _data_interp_handles[i_raster],
                        _invalid_values[i_raster],
                        is_sinc_interp);
                break;
            }
            case GDT_CFloat64: {
                geocodeRasterBlock<thrust::complex<double>>(
                        output_rasters[i_raster], input_rasters[i_raster],
                        _data_interp_handles[i_raster],
                        _invalid_values[i_raster],
                        is_sinc_interp);
                break;
            }
            case GDT_Byte: {
                geocodeRasterBlock<unsigned char>(
                        output_rasters[i_raster], input_rasters[i_raster],
                        _data_interp_handles[i_raster],
                        _invalid_values[i_raster],
                        is_sinc_interp);
                break;
            }
            case GDT_UInt32: {
                geocodeRasterBlock<unsigned int>(
                        output_rasters[i_raster], input_rasters[i_raster],
                        _data_interp_handles[i_raster],
                        _invalid_values[i_raster],
                        is_sinc_interp);
                break;
            }
            default: {
            throw isce3::except::RuntimeError(
                    ISCE_SRCINFO(), "unsupported datatype");
            }
        }
        }
    }
}

#define EXPLICIT_INSTATIATION(T)                                               \
    template void Geocode::geocodeRasterBlock<T>( \
            isce3::io::Raster& output_raster, isce3::io::Raster& input_raster, \
            const std::shared_ptr< \
                isce3::cuda::core::InterpolatorHandleVirtual>& interp_handle_ptr, \
            const std::any& invalid_value_any, const bool is_sinc_interp);

EXPLICIT_INSTATIATION(float);
EXPLICIT_INSTATIATION(thrust::complex<float>);
EXPLICIT_INSTATIATION(double);
EXPLICIT_INSTATIATION(thrust::complex<double>);
EXPLICIT_INSTATIATION(unsigned char);
EXPLICIT_INSTATIATION(unsigned int);
} // namespace isce3::cuda::geocode
