

#include <complex>
#include <Eigen/Core>
#include <limits>

#include <isce3/product/RadarGridParameters.h>

namespace isce3::image::flatten {

template<class T, int Options = Eigen::RowMajor>
using Array2D = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Options>;

template<class T, int Options = Eigen::RowMajor>
using ArrayRef2D = Eigen::Ref<Array2D<T, Options>>;

template<class T, int Options = Eigen::RowMajor>
using ArrayRefConst2D = Eigen::Ref<const Array2D<T, Options>>;

/**
 * Re-flatten a grid of SLC data from its' original grid parameters into a new set of
 * grid parameters.
 *
 * @param[out] dataBlock        The SLC data to flatten
 * @param[in] rangeIndices      range index of each coordinate pixel in the data block
 *                              in the coordinate system of the alternate radar grid
 * @param[in] radarGridOut      radar grid parameters of the alternate grid
 * @param[in] radarGridIn       radar grid parameters of the original grid
 * @param[in] inRgFirstPixel    range index of the first sample of the original grid
 * @param[in] outRgFirstPixel   range index of the first sample of the alternate grid
 */
void flattenAtCoords(
    ArrayRef2D<std::complex<float>> dataBlock,
    const ArrayRefConst2D<double> rangeIndices,
    const isce3::product::RadarGridParameters& radarGridOut,
    const isce3::product::RadarGridParameters& radarGridIn,
    const size_t inRgFirstPixel,
    const size_t outRgFirstPixel
);

} // namespace isce3::image::flatten