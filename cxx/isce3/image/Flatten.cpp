
#include "Flatten.h"

#include <isce3/product/RadarGridProduct.h>


namespace isce3::image::flatten {


/** Calculate the phase flattening parameter for a given pixel
 * 
 * This code assumes that the output and input grids have the same wavelength
 * and pixel spacing.
 * 
 * @returns     double      The flattening phase for this pixel.
 *
 * RadarGridParameters of output radar data
 * @param[in] radarGridOut
 * RadarGridParameters of input radar data
 * @param[in] radarGridIn
 * The difference between the input range index and its point on the output grid;
 * unit: range column indices (int)
 * @param[in] rangeOffset
*/
double _pixelFlatteningPhase(
    const isce3::product::RadarGridParameters& radarGridOut,
    const isce3::product::RadarGridParameters& radarGridIn,
    const double rangeOffset
)
{
    // Values from the output radar grid
    // unit: distance (meters)
    const auto startingRange = radarGridOut.startingRange();
    // unit: distance (meters) per range index position
    const auto rangePixelSpacing = radarGridOut.rangePixelSpacing();
    // unit: distance (meters) per cycle
    const auto wavelength = radarGridOut.wavelength();
    
    // Values from the input radar grid
    // unit: distance (meters)
    const auto inStartingRange = radarGridIn.startingRange();
    // The number of cycles completed by this wavelength of light in a given unit
    // distance.
    // unit: cycles per unit distance
    const auto spatialFrequency = 1. / wavelength;
    
    // The difference between the starting range and the input starting range.
    // unit: distance (meters)
    const auto startingRangeDifference = startingRange - inStartingRange;
    // The distance from the point referenced at the range pixel grid and the point that
    // its offset points to.
    // unit: distance (meters)
    const auto offsetDistance = rangeOffset * rangePixelSpacing;
    // The total distance difference between the indexed point on the output radar
    // grid and its offset point on the input radar grid.
    // unit: distance (meters)
    const auto distanceDiff = startingRangeDifference + offsetDistance;
                                                    
    // The number of cycles covered over that distance.
    // unit: cycles
    const auto cycleDiff = distanceDiff * spatialFrequency;

    // The flattening phase parameter for this pixel is the cycles in a two-way trip,
    // multiplied by 2 * pi.
    // unit: radians
    return 4. * M_PI * cycleDiff;
}


void flattenAtCoords(
    ArrayRef2D<std::complex<float>> dataBlock,
    const EArray2df64& rangeIndices,
    const isce3::product::RadarGridParameters& radarGridOut,
    const isce3::product::RadarGridParameters& radarGridIn,
    const size_t inRgFirstPixel,
    const size_t outRgFirstPixel
)
{
    const size_t outWidth = dataBlock.cols();
    const size_t outLength = dataBlock.rows();

#pragma omp parallel for collapse(2)
    for (size_t azIndexIn = 0; azIndexIn < outLength; ++azIndexIn){
        for (size_t rgIndexIn = 0; rgIndexIn < outWidth; ++rgIndexIn){

            // Get the range indices on the output grid corresponding to these indices
            // on the input grid overall.
            const double rgIndexOut = rangeIndices(azIndexIn, rgIndexIn);

            double rangeOffset = rgIndexOut - static_cast<double>(rgIndexIn) -
                static_cast<double>(outRgFirstPixel);

            const double flattenPhase = _pixelFlatteningPhase(
                radarGridOut, radarGridIn, rangeOffset
            );

            // Update dataBlock column and row from index
            const std::complex<float> cpxVal(
                std::cos(flattenPhase), std::sin(flattenPhase)
            );
            dataBlock(azIndexIn, rgIndexIn) *= cpxVal;
        }
    } // end multithreaded block
}


void getFlatteningPhase(
    ArrayRef2D<std::complex<float>> dataBlock,
    const EArray2df64& rangeIndices,
    const isce3::product::RadarGridParameters& radarGridOut,
    const isce3::product::RadarGridParameters& radarGridIn,
    const size_t inRgFirstPixel,
    const size_t outRgFirstPixel
)
{
    const size_t outWidth = dataBlock.cols();
    const size_t outLength = dataBlock.rows();

#pragma omp parallel for collapse(2)
    for (size_t azIndexIn = 0; azIndexIn < outLength; ++azIndexIn){
        for (size_t rgIndexIn = 0; rgIndexIn < outWidth; ++rgIndexIn){

            // Get the range indices on the output grid corresponding to these indices
            // on the input grid overall.
            const double rgIndexOut = rangeIndices(azIndexIn, rgIndexIn);

            double rangeOffset = rgIndexOut - static_cast<double>(rgIndexIn) -
                static_cast<double>(outRgFirstPixel);

            const double flattenPhase = _pixelFlatteningPhase(
                radarGridOut, radarGridIn, rangeOffset
            );

            // Update dataBlock column and row from index
            const std::complex<float> cpxVal(
                std::cos(flattenPhase), std::sin(flattenPhase)
            );
            dataBlock(azIndexIn, rgIndexIn) = cpxVal;
        }
    } // end multithreaded block
}


}  // namespace isce3::image::flatten
