#include <cmath>
#include <cuda_runtime.h>
#include <limits>
#include <thrust/complex.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>

#include <gdal_priv.h>
#include <pyre/journal.h>

#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Projections.h>
#include <isce3/cuda/container/RadarGeometry.h>
#include <isce3/cuda/core/OrbitView.h>
#include <isce3/cuda/core/gpuLUT2d.h>
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

#include <cstdio>
using isce3::core::Vec3;
using isce3::io::Raster;

using isce3::cuda::core::InterpolatorHandle;
using DeviceOrbitView = isce3::cuda::core::OrbitView;
using isce3::cuda::core::ProjectionBaseHandle;
using namespace isce3::geometry::detail;

template<typename T>
using DeviceInterp = isce3::cuda::core::gpuInterpolator<T>;

template<typename T>
using DeviceLUT2d = isce3::cuda::core::gpuLUT2d<T>;

namespace isce3::cuda::geocode {

/**  Coverts a batch of rows from input geogrid into radar coordinates. Outputs
 *  the pixel-space coordinates (x,y) of each resulting (range, azimuth) pair
 *  with respect to specified radargrid, as well as mask invalid pixels (out
 *  of bounds or failed to converge in geo2rdr).
 *
 * \param[out] rdr_x            pointer to device_vector of computed radar grid
 *                              x / slant range indices
 * \param[out] rdr_y            pointer to device_vector of computed radar grid
 *                              y / azimuth time indices
 * \param[out] mask             pointer to device_vector mask of valid
 *                              pixels. Mask follows numpy mask_array
 *                              convention where True is masked
 * \param[in] ellipsoid         Ellipsoid based on output geogrid coordinate
 *                              system
 * \param[in] orbit             Orbit associated with radar data
 * \param[in] dem               DEM interpolator. Maybe be of different
 *                              coordinate system than output geogrid.
 * \param[in] doppler           doppler
 * \param[in] wvl               wavelength
 * \param[in] side              look side
 * \param[in] geo2rdr_params    geo2rdr params
 * \param[in] geogrid           Geogrid defining output product/rasters
 * \param[in] radargrid         Radar grid decribing rasters to be geocoded
 * \param[in] line_start        Starting line of block
 * \param[in] block_size        Number of elements in a block
 * \param[in] proj              Projection used to covert geogrid XYZ to LLH
 *                              of output coordinate system
 */
__global__ void geoToRdrIndices(double* rdr_x, double* rdr_y, bool* mask,
        const isce3::core::Ellipsoid ellipsoid, const DeviceOrbitView orbit,
        isce3::cuda::geometry::gpuDEMInterpolator dem,
        const DeviceLUT2d<double> doppler, const double wvl,
        const isce3::core::LookSide side, const Geo2RdrParams geo2rdr_params,
        const isce3::product::GeoGridParameters geogrid,
        const RadarGridParams radargrid, const size_t line_start,
        const size_t block_size, isce3::cuda::core::ProjectionBase** proj)
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

    // geo2rdr slant range
    double r;
    // geo2rdr azimuth time initial value
    double t = radargrid.sensing_mid;

    // returns 0 if geo2rdr converges else 1
    int converged = isce3::cuda::geometry::geo2rdr(llh, ellipsoid, orbit,
            doppler, &t, &r, wvl, side, geo2rdr_params.threshold,
            geo2rdr_params.maxiter, geo2rdr_params.delta_range);

    // convert aztime and range to indices
    double y = (t - radargrid.sensing_start) * radargrid.prf;
    double x = (r - radargrid.starting_range) / radargrid.range_pixel_spacing;

    // check if indinces in bounds and set accordingly
    const bool not_in_rdr_grid =
            y < 0 || y >= radargrid.length || x < 0 || x >= radargrid.width;

    const bool invalid_index = not_in_rdr_grid || converged == 0;
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
 * pixels of current block \param[in] rdr_data_block    pointer to device vector
 * of radar data of current block \param[in] width             width of
 * rdr_data_block \param[in] length            length of rdr_data_block
 * \param[in] block_size        number of elements in a block
 * \param[in] az_1st_line       offset applied to az time indices to correctly
 *                              access current block
 * \param[in] range_1st_pixel   offset applied to slant range indices to
 * correctly access current block \param[in] invalid_value     value assigned to
 * invalid geogrid pixels \param[in] interp            interpolator used to
 * interpolate radar data to specified geogrid
 */
template<class T>
__global__ void interpolate(T* geo_data_block, const double* __restrict__ rdr_x,
        const double* __restrict__ rdr_y, const bool* __restrict__ mask,
        const T* __restrict__ rdr_data_block, const size_t width,
        const size_t length, const size_t block_size, const double az_1st_line,
        const double range_1st_pixel, const T invalid_value,
        DeviceInterp<T>** interp)
{
    // thread index (1d grid of 1d blocks)
    const auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid >= block_size)
        return;

    auto rdry = rdr_y[tid] - az_1st_line;
    auto rdrx = rdr_x[tid] - range_1st_pixel;

    // check if indices are in bounds of radar block
    // add margin to account for possibly deficient interpolator boundary checks
    constexpr double extra_margin = 4.0; // magic number for SINC_HALF
    bool out_of_bounds = rdrx < extra_margin || rdry < extra_margin ||
                         rdrx >= width - extra_margin ||
                         rdry >= length - extra_margin;

    // default to invalid value. interpolate only if in bounds and not masked.
    T interp_val = invalid_value;
    if (!(out_of_bounds || mask[tid]))
        interp_val = (*interp)->interpolate(
                rdrx, rdry, rdr_data_block, width, length);

    geo_data_block[tid] = interp_val;
}

__host__
Geocode::Geocode(const isce3::product::GeoGridParameters & geogrid,
                const isce3::container::RadarGeometry & rdr_geom,
                const Raster & dem_raster,
                const size_t lines_per_block,
                const isce3::core::dataInterpMethod data_interp_method,
                const isce3::core::dataInterpMethod dem_interp_method,
                const double threshold, const int maxiter,
                const double dr, const float invalid_value) :
    _geogrid(geogrid),
    _rdr_geom(rdr_geom),
    _ellipsoid(isce3::core::makeProjection(_geogrid.epsg())->ellipsoid()),
    _lines_per_block(lines_per_block),
    _geo_block_length(_lines_per_block),
    _n_blocks((geogrid.length() + _lines_per_block -1) / _lines_per_block),
    _az_first_line(_rdr_geom.radarGrid().length() - 1),
    _az_last_line(0),
    _range_first_pixel(_rdr_geom.radarGrid().width() - 1),
    _range_last_pixel(0),
    _dem_raster(dem_raster),
    _interp_float_handle(data_interp_method),
    _interp_cfloat_handle(data_interp_method),
    _interp_double_handle(data_interp_method),
    _interp_cdouble_handle(data_interp_method),
    _interp_unsigned_char_handle(data_interp_method),
    _interp_unsigned_int_handle(data_interp_method),
    _proj_handle(geogrid.epsg()),
    _dem_interp_method(dem_interp_method),
    _data_interp_method(data_interp_method)
{
    // init light weight radar grid
    _radar_grid.sensing_start = _rdr_geom.radarGrid().sensingStart();
    _radar_grid.sensing_mid = _rdr_geom.radarGrid().sensingMid();
    _radar_grid.prf = _rdr_geom.radarGrid().prf();
    _radar_grid.starting_range = _rdr_geom.radarGrid().startingRange();
    _radar_grid.range_pixel_spacing = _rdr_geom.radarGrid().rangePixelSpacing();
    _radar_grid.length = _rdr_geom.gridLength();
    _radar_grid.width = _rdr_geom.gridWidth();

    // Determine max number of elements per block. Last block to be processed
    // may not contain the max number of elements.
    auto n_elem = _lines_per_block * _geogrid.width();

    // Assign geo2rdr parameter values
    _geo2rdr_params.threshold = threshold;
    _geo2rdr_params.maxiter = maxiter;
    _geo2rdr_params.delta_range = dr;

    // Resize all device vectors to max block size.
    _radar_x.resize(n_elem);
    _radar_y.resize(n_elem);
    _mask.resize(n_elem);

    if (std::isnan(invalid_value)) {
        _invalid_float = std::numeric_limits<float>::quiet_NaN();
        _invalid_double = std::numeric_limits<double>::quiet_NaN();
        _invalid_unsigned_char = 255;
        _invalid_unsigned_int = 4294967295;
    } else {
        _invalid_float = invalid_value;
        _invalid_double = static_cast<double>(invalid_value);
        _invalid_unsigned_char = static_cast<unsigned char>(invalid_value);
        _invalid_unsigned_int = static_cast<unsigned int>(invalid_value);
    }
}

void Geocode::setBlockRdrCoordGrid(const size_t block_number)
{
    // make sure block index does not exceed actual number of blocks
    if (block_number >= _n_blocks) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "block number exceeds max number of blocks");
    }

    // Get block extents (of the geocoded grid)
    _line_start = block_number * _lines_per_block;

    // Set block sizes for everything but last block
    size_t block_size = _geo_block_length * _geogrid.width();
    _geo_block_length = _lines_per_block;

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
        isce3::geometry::DEMRasterToInterpolator(
            _dem_raster, _geogrid, _line_start, _geo_block_length,
            _geogrid.width(), dem_margin_in_pixels, _dem_interp_method);
    isce3::cuda::geometry::gpuDEMInterpolator dev_dem_interp(host_dem_interp);

    // copy RadarGeometry to device
    isce3::cuda::container::RadarGeometry dev_rdr_geom(_rdr_geom);

    // Create geogrid on device
    {
        const unsigned threads_per_block = 256;
        const unsigned n_blocks =
                (block_size + threads_per_block - 1) / threads_per_block;

        geoToRdrIndices<<<n_blocks, threads_per_block>>>(_radar_x.data().get(),
                _radar_y.data().get(), _mask.data().get(), _ellipsoid,
                dev_rdr_geom.orbit(), dev_dem_interp, dev_rdr_geom.doppler(),
                dev_rdr_geom.wavelength(), dev_rdr_geom.lookSide(),
                _geo2rdr_params, _geogrid, _radar_grid, _line_start, block_size,
                _proj_handle.get_proj());

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // find index of min and max values in x/range
    const auto [rdr_x_min, rdr_x_max] = masked_minmax(_radar_x, _mask);

    _range_first_pixel = std::min(_rdr_geom.radarGrid().width() - 1,
            static_cast<size_t>(std::floor(rdr_x_min)));

    _range_last_pixel = std::max(static_cast<size_t>(0),
            static_cast<size_t>(std::ceil(rdr_x_max) - 1));

    // find index of min and max values in y/azimuth
    const auto [rdr_y_min, rdr_y_max] = masked_minmax(_radar_y, _mask);

    _az_first_line = std::min(_rdr_geom.radarGrid().length() - 1,
            static_cast<size_t>(std::floor(rdr_y_min)));

    _az_last_line = std::max(static_cast<size_t>(0),
            static_cast<size_t>(std::ceil(rdr_y_max) - 1));

    // Extra margin for interpolation to avoid gaps between blocks in output
    const size_t interp_margin {5};
    _range_first_pixel = interp_margin > _range_first_pixel ? 0 : _range_first_pixel - interp_margin;
    _range_last_pixel = std::min(_rdr_geom.radarGrid().width() - 1,
                             _range_last_pixel + interp_margin);

    _az_first_line = interp_margin > _az_first_line ? 0 : _az_first_line - interp_margin;
    _az_last_line = std::min(_rdr_geom.radarGrid().length() - 1,
                             _az_last_line+interp_margin);

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

void Geocode::rasterDtypeInterpCheck(const int dtype) const
{
    if ((dtype == GDT_Byte || dtype == GDT_UInt32) &&
        _data_interp_method != isce3::core::NEAREST_METHOD)
    {
        std::string err_str {"int type of raster can only use nearest neighbor interp"};
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), err_str);
    }
}

template<class T>
void Geocode::geocodeRasterBlock(Raster& output_raster, Raster& input_raster)
{
    rasterDtypeInterpCheck(input_raster.dtype());

    // determine number of elements in output vector
    const auto n_elem_out = _geo_block_length * _geogrid.width();

    // determine by type, interp and invalid value
    DeviceInterp<T>** interp;
    T invalid_value;
    if constexpr (std::is_same_v<T, float>) {
        interp = _interp_float_handle.getInterp();
        invalid_value = _invalid_float;
    } else if constexpr (std::is_same_v<T, thrust::complex<float>>) {
        interp = _interp_cfloat_handle.getInterp();
        invalid_value = thrust::complex<float>(_invalid_float, _invalid_float);
    } else if constexpr (std::is_same_v<T, double>) {
        interp = _interp_double_handle.getInterp();
        invalid_value = _invalid_double;
    } else if constexpr (std::is_same_v<T, thrust::complex<double>>) {
        interp = _interp_cdouble_handle.getInterp();
        invalid_value = thrust::complex<double>(_invalid_double,
                                                _invalid_double);
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        interp = _interp_unsigned_char_handle.getInterp();
        invalid_value = _invalid_unsigned_char;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        interp = _interp_unsigned_int_handle.getInterp();
        invalid_value = _invalid_unsigned_int;
    }

    // 0 width indicates current block is out of bounds
    if (_rdr_block_width == 0) {
        // set entire block to invalid value
        thrust::host_vector<T> h_geo_data_block =
                thrust::host_vector<T>(n_elem_out, invalid_value);
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
    thrust::device_vector<T> d_geo_data_block(n_elem_out);

    // Perform interpolation on device
    {
        const unsigned threads_per_block = 256;
        const unsigned n_blocks =
                (n_elem_out + threads_per_block - 1) / threads_per_block;
        interpolate<<<n_blocks, threads_per_block>>>(
                d_geo_data_block.data().get(), _radar_x.data().get(),
                _radar_y.data().get(), _mask.data().get(),
                d_rdr_data_block.data().get(), _rdr_block_width,
                _rdr_block_length, n_elem_out,
                static_cast<double>(_az_first_line),
                static_cast<double>(_range_first_pixel), invalid_value, interp);

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // set output raster block from device
    thrust::host_vector<T> h_geo_data_block = d_geo_data_block;

    output_raster.setBlock(&h_geo_data_block[0], 0, _line_start,
            _geogrid.width(), _geo_block_length, 1);
}

void Geocode::geocodeRasters(
        std::vector<std::reference_wrapper<isce3::io::Raster>> output_rasters,
        std::vector<std::reference_wrapper<isce3::io::Raster>> input_rasters)
{
    // check if vectors are of same length
    if (output_rasters.size() != input_rasters.size()) {
        throw isce3::except::LengthError(
                ISCE_SRCINFO(), "number of input and output rasters not equal");
    }

    const auto n_raster_pairs = output_rasters.size();

    // check if raster types consistent with data interp method
    for (size_t i_raster = 0; i_raster < n_raster_pairs; ++i_raster)
        rasterDtypeInterpCheck(input_rasters[i_raster].get().dtype());

    // iterate over blocks
    for (size_t i_block = 0; i_block < _n_blocks; ++i_block) {

        // set radar coords for each geocode obj for curret block
        setBlockRdrCoordGrid(i_block);

        for (size_t i_raster = 0; i_raster < n_raster_pairs; ++i_raster)
        {
            const int dtype = input_rasters[i_raster].get().dtype();
            switch (dtype) {
                case GDT_Float32:   {
                    geocodeRasterBlock<float>(
                            output_rasters[i_raster], input_rasters[i_raster]);
                    break; }
                case GDT_CFloat32:  {
                    geocodeRasterBlock<thrust::complex<float>>(
                            output_rasters[i_raster], input_rasters[i_raster]);
                    break;}
                case GDT_Float64:   {
                    geocodeRasterBlock<double>(
                            output_rasters[i_raster], input_rasters[i_raster]);
                    break; }
                case GDT_CFloat64:  {
                    geocodeRasterBlock<thrust::complex<double>>(
                            output_rasters[i_raster], input_rasters[i_raster]);
                    break;}
                case GDT_Byte:  {
                    geocodeRasterBlock<unsigned char>(
                            output_rasters[i_raster], input_rasters[i_raster]);
                    break;}
                case GDT_UInt32:  {
                    geocodeRasterBlock<unsigned int>(
                            output_rasters[i_raster], input_rasters[i_raster]);
                    break;}
                default: {
                    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                            "unsupported datatype");
                         }
            }
        }
    }
}

#define EXPLICIT_INSTATIATION(T)                                               \
    template void Geocode::geocodeRasterBlock<T>(                              \
            Raster & output_raster, Raster & input_raster);
EXPLICIT_INSTATIATION(float);
EXPLICIT_INSTATIATION(thrust::complex<float>);
EXPLICIT_INSTATIATION(double);
EXPLICIT_INSTATIATION(thrust::complex<double>);
EXPLICIT_INSTATIATION(unsigned char);
EXPLICIT_INSTATIATION(unsigned int);
} // namespace isce3::cuda::geocode
