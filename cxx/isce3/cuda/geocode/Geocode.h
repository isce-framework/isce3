#pragma once

#include <any>
#include <functional>
#include <memory>
#include <vector>

#include <isce3/core/forward.h>
#include <isce3/geometry/forward.h>

#include <thrust/device_vector.h>

#include <isce3/container/RadarGeometry.h>
#include <isce3/cuda/container/forward.h>
#include <isce3/cuda/core/InterpolatorHandle.h>
#include <isce3/cuda/core/ProjectionBaseHandle.h>
#include <isce3/cuda/core/gpuInterpolator.h>
#include <isce3/cuda/core/gpuLUT2d.h>
#include <isce3/cuda/core/gpuProjections.h>
#include <isce3/cuda/product/SubSwaths.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>

namespace isce3::cuda::geocode {

template<typename T>
using DeviceLUT2d = isce3::cuda::core::gpuLUT2d<T>;

/* light weight radar grid container */
struct RadarGridParamsLite {
    double sensing_start;
    double sensing_mid;
    double prf;
    double starting_range;
    double range_pixel_spacing;
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
     * \param[in] lines_per_block       Number of lines to be processed per block
     */
    Geocode(const isce3::product::GeoGridParameters & geogrid,
            const isce3::container::RadarGeometry & rdr_geom,
            const size_t lines_per_block = 1000);

    /** Geocode rasters with a shared geogrid. Block processing handled
     * internally in function.
     *
     * \param[in] output_rasters    Geocoded rasters
     * \param[in] input_rasters     Rasters to be geocoded. Needs to be same
     *                              size as output_rasters.
     * \param[in] interp_methods        Data interpolation method per raster
     * \param[in] raster_datatypes      GDAL type of each raster represented by
     *                                  equivalent int. Necessary to correctly
     *                                  initialize templated interpolators.
     * \param[in] invalid_values        Invalid values for each raster
     * \param[in] dem_raster            DEM used to calculate radar grid indices
     * \param[in] hostNativeDoppler     Doppler centroid of data in Hz associated
     *                                  radar grid, as a function azimuth and
     *                                  range
     * \param[in] hostAzTimeCorrection  geo2rdr azimuth additive correction, in
     *                                  seconds, as a function of azimuth and
     *                                  range
     * \param[in] hostSRangeCorrection  geo2rdr slant range additive
     *                                  correction, in seconds, as a function
     *                                  of azimuth and range
     * \param[in]  subswaths            subswath mask representing valid
     *                                  portions of a swath
     * \param[in] dem_interp_method     DEMinterpolation method
     * \param[in] threshold             Convergence threshold for geo2rdr
     * \param[in] maxiter               Maximum iterations for geo2rdr
     * \param[in] dr                    Step size for numerical gradient for
     *                                  geo2rdr
     */
    void geocodeRasters(
            std::vector<std::reference_wrapper<isce3::io::Raster>>& output_rasters,
            std::vector<std::reference_wrapper<isce3::io::Raster>>& input_rasters,
            const std::vector<isce3::core::dataInterpMethod>& interp_methods,
            const std::vector<GDALDataType>& raster_datatypes,
            const std::vector<double>& invalid_values,
            Raster& dem_raster,
            const isce3::core::LUT2d<double>& hostNativeDoppler = {},
            const isce3::core::LUT2d<double>& hostAzTimeCorrection = {},
            const isce3::core::LUT2d<double>& hostSRangeCorrection = {},
            const isce3::product::SubSwaths* subswaths = nullptr,
            const isce3::core::dataInterpMethod dem_interp_method =
                    isce3::core::BIQUINTIC_METHOD,
            const double threshold = 1e-8,
            const int maxiter = 50,
            const double dr = 10);

private:
    // total number of blocks necessary to geocoding a provided geogrid
    size_t _n_blocks;

    // geogrid defining output product
    isce3::product::GeoGridParameters _geogrid;

    // light weight clone of isce3::product::RadarGridParameters _radar_grid
    RadarGridParamsLite _radar_grid;

    // radar geometry describing input rasters
    isce3::container::RadarGeometry _rdr_geom;

    // ellipsoid based on EPSG of output grid
    isce3::core::Ellipsoid _ellipsoid;

    // Radar grid indices of block number last passed to setBlockRdrCoordGrid
    thrust::device_vector<double> _radar_x;
    thrust::device_vector<double> _radar_y;

    // Valid pixel map; follows numpy masked array convention
    thrust::device_vector<bool> _mask;

    // DEM used to calculate radar grid indices
    //isce3::io::Raster _dem_raster;

    // Radar grid boundaries of block last passed to setBlockRdrCoordGrid,
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

    /* Check if input and output raster are consistent in:
     * 1. size - same number input and output rasters
     * 2. type - input and output rasters datatypes match expected values set
     *           in the constructor
     *
     * \param[in] output_rasters    Geocoded rasters
     * \param[in] input_rasters     Rasters to be geocoded. Needs to be same
     *                              size as output_rasters.
     * \param[in] interp_methods        Data interpolation method per raster
     * \param[in] raster_datatypes      GDAL type of each raster represented by
     *                                  equivalent int. Necessary to correctly
     *                                  initialize templated interpolators.
     * \param[in] invalid_values_double Invalid values for each raster as double, to
                                        be converted later to the correct type
     */
    void ensureRasterConsistency(
            const std::vector<std::reference_wrapper<isce3::io::Raster>>& output_rasters,
            const std::vector<std::reference_wrapper<isce3::io::Raster>>& input_rasters,
            const std::vector<isce3::core::dataInterpMethod>& interp_methods,
            const std::vector<GDALDataType>& raster_datatypes,
            const std::vector<double>& invalid_values_double) const;

    /** Calculate and set radar grid coordinates of geocode grid for a given
     *  block number in device memory. The radar grid coordinates can then be
     *  repeatedly used by geocodeRasterBlock for geocoding rasters of given
     *  block number.
     *
     * \param[in] block_number      Index of block where radar grid coordinates
     *                              coordinates will be calculated and set
     * \param[in] dem_raster        DEM used to calculate radar grid indices
     * \param[in] dem_interp_method     DEMinterpolation method
     * \param[in] dev_rdr_geom          Radar geometry describing input raster
     *                                  stored on-device
     * \param[in] geo2rdr_params        Parameters used by geo2rdr to compute
     *                                  geogrids radar grid coordinates
     * \param[in] nativeDoppler         Doppler centroid of data in Hz associated
     *                                  radar grid, as a function azimuth and
     *                                  range
     * \param[in] azTimeCorrection      geo2rdr azimuth additive correction, in
     *                                  seconds, as a function of azimuth and
     *                                  range
     * \param[in] sRangeCorrection      geo2rdr slant range additive
     *                                  correction, in seconds, as a function
     *                                  of azimuth and range
     * \param[in]  subswaths            subswath mask representing valid
     *                                  portions of a swath
     */
    void setBlockRdrCoordGrid(const size_t block_number,
            Raster& dem_raster,
            const isce3::core::dataInterpMethod dem_interp_method,
            const isce3::cuda::container::RadarGeometry& dev_rdr_geom,
            const isce3::geometry::detail::Geo2RdrParams geo2rdr_params,
            const DeviceLUT2d<double>& nativeDoppler,
            const DeviceLUT2d<double>& azTimeCorrection,
            const DeviceLUT2d<double>& sRangeCorrection,
            const isce3::cuda::product::ViewSubSwaths& subswaths);

    /** Geocode a block of raster according to grid last set in
     *  setBlockRdrCoordGrid. Only operates on 1st raster band of input/output
     *  dataset.
     *
     * \param[in] output_raster     Geocoded individual raster
     * \param[in] input_raster      Individual raster to be geocoded
     * \param[in] interp_handle_ptr Interpolator handle containing device
     *                              interpolator to be used on current block
     * \param[in] invalid_value_any Invalid value to initialize block with - to
     *                              be casted to template type.
     * \param[in] is_sinc_interp    True if sinc interpolation is to be used.
     *                              Allows the kernel to call sinc interpolator
     *                              differently from other interpolators.
     */
    template<typename T>
    void geocodeRasterBlock(
            isce3::io::Raster& output_raster, isce3::io::Raster& input_raster,
            const std::shared_ptr<
                isce3::cuda::core::InterpolatorHandleVirtual>& interp_handle_ptr,
            const std::any& invalid_value_any, const bool is_sinc_interp);
};
} // namespace isce3::cuda::geocode
