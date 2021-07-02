#pragma once

#include <isce3/core/forward.h>
#include <isce3/geometry/forward.h>

#include <thrust/device_vector.h>

#include <isce3/container/RadarGeometry.h>
#include <isce3/cuda/core/InterpolatorHandle.h>
#include <isce3/cuda/core/ProjectionBaseHandle.h>
#include <isce3/cuda/core/gpuInterpolator.h>
#include <isce3/cuda/core/gpuProjections.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>

namespace isce3::cuda::geocode {

/* light weight radar grid container */
struct RadarGridParams {
    double sensing_start;
    double sensing_mid;
    double prf;
    double starting_range;
    double range_pxl_spacing;
    size_t length;
    size_t width;
};

/** Class containing functions for geocoding by interpolation with CUDA.
 *
 * As designed, a single object can be used to geocode several rasters that
 * share a common radar grid to a common geogrid. To allow processing on
 * limited device memory, a geogrid is broken up into uniform blocks and
 * geocoding is performed per block over several rasters. The logic and control
 * of iterating over blocks and rasters is not included in this class. ISCE3
 * will contain a Python module to perform this.
 *
 * Python-ish pseudo code for geocoding a several rasters:
 * import isce3
 * # instantiate obj
 * geocode_obj = isce3.cuda.geocode(number_of_lines_per_block, ...)
 * # prepare input & output rasters
 * output_rasters = [isce3.io.Raster(x) for x in list_output_rasters]
 * input_rasters = [isce3.io.Raster(x) for x in list_input_rasters]
 * # iterate over blocks
 * for block in range(geocode_obj.number_of_blocks):
 *     # set radar grid coords for current block
 *     geocode_obj.setBlockRdrCoordGrid(block)
 *     # iterate over rasters to geocode to current block
 *     for output_raster, input_raster in zip(output_rasters, input_rasters):
 *         geocode_obj.geocodeRasterBlock(output_raster, input_raster)
 *
 */
class Geocode {
public:
    /** Class constructor. Sets values to be shared by all blocks. Also
     *  calculates number of blocks needed to completely geocode provided
     *  geogrid.
     *
     * \param[in] geogrid               Geogrid defining output product
     * \param[in] rdr_geom              Radar geometry describing input rasters
     * \param[in] dem_raster            DEM used to calculate radar grid indices
     * \param[in] dem_margin            Extra margin applied to bounding box to
     *                                  load DEM. Units are need to match
     * geogrid EPSG units. \param[in] lines_per_block       Number of lines to
     * be processed per block \param[in] data_interp_method    Data
     * interpolation method \param[in] dem_interp_method     DEMinterpolation
     * method \param[in] threshold             Convergence threshold for geo2rdr
     * \param[in] maxiter               Maximum iterations for geo2rdr
     * \param[in] dr                    Step size for numerical gradient for
     *                                  geo2rdr
     * \param[in] invalid_value         Value assigned to invalid geogrid pixels
     */
    Geocode(const isce3::product::GeoGridParameters& geogrid,
            const isce3::container::RadarGeometry& rdr_geom,
            const isce3::io::Raster& dem_raster, const double dem_margin,
            const size_t lines_per_block = 1000,
            const isce3::core::dataInterpMethod data_interp_method =
                    isce3::core::BILINEAR_METHOD,
            const isce3::core::dataInterpMethod dem_interp_method =
                    isce3::core::BIQUINTIC_METHOD,
            const double threshold = 1e-8, const int maxiter = 50,
            const double dr = 10, const float invalid_value = 0.0);

    /** Calculate and set radar grid coordinates of geocode grid for a given
     *  block number in device memory. The radar grid coordinates can then be
     *  repeatedly used by geocodeRasterBlock for geocoding rasters of given
     *  block number.
     *
     * \param[in] block_number      Index of block where radar grid coordinates
     *                              coordinates will be calculated and set
     */
    void setBlockRdrCoordGrid(const size_t block_number);

    /** Geocode a block of raster according to grid last set in
     *  setBlockRdrCoordGrid. Only operates on 1st raster band of input/output
     *  dataset.
     *
     * \param[in] output_raster     Geocoded individual raster
     * \param[in] input_raster      Individual raster to be geocoded
     */
    template<typename T>
    void geocodeRasterBlock(
            isce3::io::Raster& output_raster, isce3::io::Raster& input_raster);

    size_t numBlocks() const { return _n_blocks; }
    size_t linesPerBlock() const { return _lines_per_block; }

private:
    // number of lines to be processed in a block
    size_t _lines_per_block;

    // total number of blocks necessary to geocoding a provided geogrid
    size_t _n_blocks;

    // geogrid defining output product
    isce3::product::GeoGridParameters _geogrid;

    // radar grid describing input rasters
    RadarGridParams _radar_grid;

    // light weight clone of isce3::product::RadarGridParams _radar_grid
    isce3::container::RadarGeometry _rdr_geom;

    // ellipsoid based on EPSG of output grid
    isce3::core::Ellipsoid _ellipsoid;

    // geo2rdr params used in radar index calculation
    isce3::geometry::detail::Geo2RdrParams _geo2rdr_params;

    // Radar grid indices of block number last passed to setBlockRdrCoordGrid
    thrust::device_vector<double> _radar_x;
    thrust::device_vector<double> _radar_y;

    // Valid pixel map; follows numpy masked array convention
    thrust::device_vector<bool> _mask;

    // DEM used to calculate radar grid indices
    isce3::io::Raster _dem_raster;

    // Extra margin for the dem relative to the geocoded grid
    double _dem_margin;

    // radar grid boundaries of block last passed to setBlockRdrCoordGrid,
    // not for entire geogrid
    size_t _az_first_line;
    size_t _az_last_line;
    size_t _range_first_pixel;
    size_t _range_last_pixel;
    size_t _geo_block_length;
    size_t _rdr_block_length;
    size_t _rdr_block_width;
    size_t _line_start;

    // projection based on geogrid EPSG - common to all blocks
    isce3::cuda::core::ProjectionBaseHandle _proj_handle;

    // interpolator used to geocode - common to all blocks
    isce3::cuda::core::InterpolatorHandle<float> _interp_float_handle;
    isce3::cuda::core::InterpolatorHandle<thrust::complex<float>>
            _interp_cfloat_handle;
    isce3::cuda::core::InterpolatorHandle<double> _interp_double_handle;
    isce3::cuda::core::InterpolatorHandle<thrust::complex<double>>
            _interp_cdouble_handle;
    isce3::cuda::core::InterpolatorHandle<unsigned char>
            _interp_unsigned_char_handle;

    // value applied to invalid geogrid pixels
    float _invalid_float;
    double _invalid_double;
    unsigned char _invalid_unsigned_char;

    isce3::core::dataInterpMethod _dem_interp_method;
};
} // namespace isce3::cuda::geocode
