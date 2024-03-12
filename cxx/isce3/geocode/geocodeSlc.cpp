#include "geocodeSlc.h"

#include <algorithm>
#include <memory>
#include <tuple>

#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Poly2d.h>
#include <isce3/core/Projections.h>
#include <isce3/geocode/baseband.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/loadDem.h>
#include <isce3/geometry/geometry.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/product/RadarGridProduct.h>
#include <isce3/product/RadarGridParameters.h>

namespace isce3::geocode {

/**
 * Remove range and azimuth phase carrier from a block of input radar SLC data
 *
 * @param[out] rdrDataBlock     block of input SLC data in radar grid to be deramped
 * @tparam[in] azCarrierPhase   azimuth carrier phase of the SLC data, in radian, as a function of azimuth and range
 * @tparam[in] rgCarrierPhase   range carrier phase of the SLC data, in radian, as a function of azimuth and range
 * @param[in] azimuthFirstLine  line index of the first sample of the block of input data with respect to the origin of the full SLC scene
 * @param[in] rangeFirstPixel   pixel index of the first sample of the block of input data with respect to the origin of the full SLC scene
 * @param[in] radarGrid         radar grid parameters of radar data
 */
template <typename AzRgFunc>
void carrierPhaseDeramp(
        Eigen::Ref<isce3::core::EArray2D<std::complex<float>>> rdrDataBlock,
        const AzRgFunc& azCarrierPhase, const AzRgFunc& rgCarrierPhase,
        const size_t azimuthFirstLine, const size_t rangeFirstPixel,
        const isce3::product::RadarGridParameters& radarGrid)
{
    const size_t rdrBlockLength = rdrDataBlock.rows();
    const size_t rdrBlockWidth = rdrDataBlock.cols();

    // remove carrier from radar data
#pragma omp parallel for
    for (size_t ii = 0; ii < rdrBlockLength * rdrBlockWidth; ++ii) {
        auto i = ii / rdrBlockWidth;
        auto j = ii % rdrBlockWidth;

        // Offset for block starting line
        const auto i_az = i + azimuthFirstLine;
        const double az = radarGrid.sensingStart() + i_az / radarGrid.prf();

        // Offset for block starting pixel
        const auto j_rg = j + rangeFirstPixel;
        const double rg = radarGrid.startingRange() +
                j_rg * radarGrid.rangePixelSpacing();

        // Evaluate the pixel's carrier phase
        const float carrierPhase = rgCarrierPhase.eval(az, rg)
                + azCarrierPhase.eval(az, rg);

        // Remove carrier at current radar grid indices
        const std::complex<float> cpxVal(std::cos(carrierPhase),
                                         -std::sin(carrierPhase));
        rdrDataBlock(i, j) *= cpxVal;
    }
}


/**
 * Add back range and azimuth phase carrier and simultaneously flatten the block of geocoded SLC
 *
 * @param[out] geoDataBlock     geocoded SLC data whose phase will be added by carrier phase and flatten by geometrical phase
 * @param[out] carrierPhaseBlock     geocoded carrier phase data that could have been added to SLC data
 * @param[out] flattenPhaseBlock     geocoded flattening phase data that could have been flatten to SLC data
 * @param[in] rdrDataBlock      radar grid SLC data
 * @tparam[in] azCarrierPhase   azimuth carrier phase of the SLC data, in radian, as a function of azimuth and range
 * @tparam[in] rgCarrierPhase   range carrier phase of the SLC data, in radian, as a function of azimuth and range
 * @param[in] nativeDopplerLUT  native doppler of SLC image
 * @param[in] rangeIndices      range (radar-coordinates x) index of the pixels in geo-grid
 * @param[in] azimuthIndices    azimuth (radar-coordinates y) index of the pixels in geo-grid
 * @param[in] radarGrid         radar grid parameters
 * @param[in] flatten           flag to flatten the geocoded SLC
 * @param[in] reramp            flag to reramp the geocoded SLC
 * @param[in] azimuthFirstLine  line index of the first sample of the block
 * @param[in] rangeFirstPixel   pixel index of the first sample of the block
 * @param[in] flattenWithCorrectedSRng  flag to use corrected slant range for flattening
 * @param[in] uncorrectedSRngs  slant range without correction, in meters, indexed
 *                              by geo-grid indices
 */
template <typename AzRgFunc>
void carrierPhaseRerampAndFlatten(
        EArray2dc64 geoDataBlock,
        EArray2df64 carrierPhaseBlock,
        EArray2df64 flattenPhaseBlock,
        const EArray2dc64 rdrDataBlock,
        const AzRgFunc& azCarrierPhase, const AzRgFunc& rgCarrierPhase,
        const isce3::core::LUT2d<double>& nativeDopplerLUT,
        isce3::core::Matrix<double>& rangeIndices,
        isce3::core::Matrix<double>& azimuthIndices,
        const isce3::product::RadarGridParameters& radarGrid,
        const bool flatten, const bool reramp, const size_t azimuthFirstLine,
        const size_t rangeFirstPixel, const bool flattenWithCorrectedSRng,
        isce3::core::Matrix<double>& uncorrectedSRngs)
{
    const size_t outWidth = geoDataBlock.cols();
    const size_t outLength = geoDataBlock.rows();
    const int inWidth = rdrDataBlock.cols();
    const int inLength = rdrDataBlock.rows();
    const int chipHalf = isce3::core::SINC_ONE / 2;

#pragma omp parallel for
    for (size_t ii = 0; ii < outLength * outWidth; ++ii) {
        auto i = ii / outWidth;
        auto j = ii % outWidth;

        // Get double and int rg/az coordinates (accounting for block offsets)
        const double RgIndex = rangeIndices(i,j) - rangeFirstPixel;
        const double AzIndex = azimuthIndices(i,j) - azimuthFirstLine;

        // Truncate rg/az indices to int
        const int intRgIndex = static_cast<int>(RgIndex);
        const int intAzIndex = static_cast<int>(AzIndex);

        // Check if chip indices could be outside radar grid (plus interp chip)
        // i.e. Check if current pixel was interpolated
        // Skip if chip indices outside int value of radar grid
        if ((intRgIndex < chipHalf) || (intRgIndex >= (inWidth - chipHalf)))
            continue;
        if ((intAzIndex < chipHalf) || (intAzIndex >= (inLength - chipHalf)))
            continue;

        // Slant Range at the current output pixel
        const double rng = radarGrid.startingRange() +
                rangeIndices(i,j) * radarGrid.rangePixelSpacing();

        // Azimuth time at the current output pixel
        const double az = radarGrid.sensingStart() +
                          azimuthIndices(i,j) / radarGrid.prf();

        // Skip pixel if doppler could not be evaluated
        if (not nativeDopplerLUT.contains(az, rng))
            continue;

        // Evaluate range and azimuth carriers
        const double carrierPhase =
            rgCarrierPhase.eval(az, rng) + azCarrierPhase.eval(az, rng);

        // Compute flatten phase based on corrected or uncorrected slant range
        const double sRngFlatten = flattenWithCorrectedSRng ?
            rng : uncorrectedSRngs(i, j);
        const double flattenPhase =
            4.0 * (M_PI / radarGrid.wavelength()) * sRngFlatten;

        // Add all the phases together as needed
        double totalPhase = 0.0;
        if (reramp)
            totalPhase += carrierPhase;
        if (flatten)
            totalPhase += flattenPhase;

        // assign values to carrier and flattening phase arrays if shape
        // matches geoDataBlock shape
        if (carrierPhaseBlock.rows() == geoDataBlock.rows() and
                carrierPhaseBlock.cols() == geoDataBlock.cols())
            carrierPhaseBlock(i, j) = carrierPhase;

        if (flattenPhaseBlock.rows() == geoDataBlock.rows() and
                flattenPhaseBlock.cols() == geoDataBlock.cols())
            flattenPhaseBlock(i, j) = flattenPhase;

        // Update geoDataBlock column and row from index
        const std::complex<float> cpxVal(std::cos(totalPhase),
                                         std::sin(totalPhase));
        geoDataBlock(i, j) *= cpxVal;
    }
}


/** Interpolate radar data block to geo data block
 *
 * @param[in] rdrDataBlock      block of SLC data in radar coordinates basebanded in range direction
 * @param[out] geoDataBlock     block of data in geo coordinates
 * @param[in] rangeIndices      range (radar-coordinates x) index of the pixels in geo-grid
 * @param[in] azimuthIndices    azimuth (radar-coordinates y) index of the pixels in geo-grid
 * @param[in] azimuthFirstLine  line index of the first sample of the block
 * @param[in] rangeFirstPixel   pixel index of the first sample of the block
 * @param[in] sincInterp        sinc interpolator object
 * @param[in] radarGrid         RadarGridParameters of radar data
 * @param[in] nativeDopplerLUT  native doppler of SLC image
 */
void interpolate(
        const Eigen::Ref<isce3::core::EArray2D<std::complex<float>>> rdrDataBlock,
        Eigen::Ref<isce3::core::EArray2D<std::complex<float>>> geoDataBlock,
        isce3::core::Matrix<double>& rangeIndices,
        isce3::core::Matrix<double>& azimuthIndices,
        const int azimuthFirstLine, const int rangeFirstPixel,
        const isce3::core::Interpolator<std::complex<float>>* sincInterp,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::LUT2d<double>& nativeDopplerLUT)
{
    const int chipSize = isce3::core::SINC_ONE;
    const int outWidth = geoDataBlock.cols();
    const int outLength = geoDataBlock.rows();
    const int inWidth = rdrDataBlock.cols();
    const int inLength = rdrDataBlock.rows();
    const int chipHalf = isce3::core::SINC_HALF;

#pragma omp parallel for
    for (size_t ii = 0; ii < outLength * outWidth; ++ii) {
        auto i = ii / outWidth;
        auto j = ii % outWidth;

        // adjust the row and column indicies for the current block,
        // i.e., moving the origin to the top-left of this radar block.
        double RgIndex = rangeIndices(i,j) - rangeFirstPixel;
        double AzIndex = azimuthIndices(i,j) - azimuthFirstLine;

        // Truncate rg/az coordinates to int
        const int intRgIndex = static_cast<int>(RgIndex);
        const int intAzIndex = static_cast<int>(AzIndex);

        // Save the fractional parts of rg/az coordinates
        const double fracRgIndex = RgIndex - intRgIndex;
        const double fracAzIndex = AzIndex - intAzIndex;

        // Check if chip indices could be outside radar grid
        // Skip if chip indices out of bounds
        if ((intRgIndex < chipHalf) || (intRgIndex >= (inWidth - chipHalf)))
            continue;
        if ((intAzIndex < chipHalf) || (intAzIndex >= (inLength - chipHalf)))
            continue;

        // Slant Range at the current output pixel
        const double rng =
                radarGrid.startingRange() +
                rangeIndices(i,j) * radarGrid.rangePixelSpacing();

        // Azimuth time at the current output pixel
        const double az = radarGrid.sensingStart() +
                          azimuthIndices(i,j) / radarGrid.prf();

        if (not nativeDopplerLUT.contains(az, rng))
            continue;

        // Evaluate doppler at current range and azimuth time
        const double doppFreq =
                nativeDopplerLUT.eval(az, rng) * 2 * M_PI / radarGrid.prf();

        isce3::core::Matrix<std::complex<float>> chip(chipSize, chipSize);
        // Read data chip
        for (int ii = 0; ii < chipSize; ++ii) {
            // Row to read from
            const int chipRow = intAzIndex + ii - chipHalf;

            // Compute doppler frequency at current row
            const double doppPhase = doppFreq * (ii - chipHalf);
            const std::complex<float> doppVal(std::cos(doppPhase),
                                              -std::sin(doppPhase));

            for (int jj = 0; jj < chipSize; ++jj) {
                // Column to read from
                const int chipCol = intRgIndex + jj - chipHalf;

                // Set the data values after doppler demodulation
                chip(ii, jj) = rdrDataBlock(chipRow, chipCol) * doppVal;
            }
        }

        // Interpolate chip
        const std::complex<float> cval =
                sincInterp->interpolate(chipHalf + fracRgIndex,
                        chipHalf + fracAzIndex, chip);

        // Compute doppler that was demodulated from chip in interpolation to
        // be added back
        const auto azLocation = fracAzIndex;
        const auto doppFreqToAddBack = doppFreq * azLocation;
        const std::complex<float> doppValAddBack(std::cos(doppFreqToAddBack),
                std::sin(doppFreqToAddBack));

        // Set geoDataBlock column and row from index with doppler added back
        geoDataBlock(i, j) = cval * doppValAddBack;
    }
}


template<typename AzRgFunc>
void geocodeSlc(
        isce3::io::Raster& outputRaster, isce3::io::Raster& inputRaster,
        isce3::io::Raster& demRaster,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::product::GeoGridParameters& geoGrid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& nativeDoppler,
        const isce3::core::LUT2d<double>& imageGridDoppler,
        const isce3::core::Ellipsoid& ellipsoid, const double& thresholdGeo2rdr,
        const int& numiterGeo2rdr, const size_t& linesPerBlock,
        const bool flatten, const bool reramp,
        const AzRgFunc& azCarrierPhase, const AzRgFunc& rgCarrierPhase,
        const isce3::core::LUT2d<double>& azTimeCorrection,
        const isce3::core::LUT2d<double>& sRangeCorrection,
        const bool flattenWithCorrectedSRng,
        const std::complex<float> invalidValue,
        isce3::io::Raster* carrierPhaseRaster,
        isce3::io::Raster* flattenPhaseRaster)
{
    geocodeSlc(outputRaster, inputRaster, demRaster, radarGrid, radarGrid,
            geoGrid, orbit,nativeDoppler, imageGridDoppler, ellipsoid,
            thresholdGeo2rdr, numiterGeo2rdr, linesPerBlock,
            flatten, reramp, azCarrierPhase, rgCarrierPhase, azTimeCorrection,
            sRangeCorrection, flattenWithCorrectedSRng, invalidValue,
            carrierPhaseRaster, flattenPhaseRaster);
}


/**
 * Ensure sliced radar grid fits within full radar grid
 * @param[in] fullGrid      radar grid to be validated against
 * @param[in] slicedGrid    sliced radar grid to be validated
 */
inline void
validate_slice(const isce3::product::RadarGridParameters& fullGrid,
        const isce3::product::RadarGridParameters& slicedGrid) {
    if (slicedGrid.sensingStart() < fullGrid.sensingStart()) {
        std::string error_msg("sliced grid sensing start < full grid sensing start");
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    if (slicedGrid.sensingStop() > fullGrid.sensingStop()) {
        std::string error_msg("sliced grid sensing stop < full grid sensing stop");
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    if (slicedGrid.startingRange() < fullGrid.startingRange()) {
        std::string error_msg("sliced grid staring range < full grid staring range");
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    if (slicedGrid.endingRange() > fullGrid.endingRange()) {
        std::string error_msg("sliced grid ending range < full grid ending range");
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    if (slicedGrid.lookSide() != fullGrid.lookSide()) {
        std::string error_msg("sliced grid look side != full grid look side");
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    if (slicedGrid.wavelength() != fullGrid.wavelength()) {
        std::string error_msg("sliced grid wavelength != full grid wavelength");
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
}


/**
 * Check if arrays in a vector all have the same size
 * @param[out] bool     true if all arrays have same size, else false
 * @param[in]  arrays   vector of arrays
 */
inline bool
array_sizes_consistent(const std::vector<EArray2dc64>& arrays) {
    // Returns true if the two input arrays have unequal shapes.
    auto unequal_array_dims = [](const auto& arr1, const auto& arr2) -> bool {
        return (arr1.rows() != arr2.rows()) or (arr1.cols() != arr2.cols());
    };

    // Check that no adjacent pair of arrays had unequal dimensions.
    const auto res = std::adjacent_find(
        arrays.begin(),
        arrays.end(),
        unequal_array_dims
    );
    return res == arrays.end();
}


template<typename AzRgFunc>
void geocodeSlc(
        isce3::io::Raster& outputRaster, isce3::io::Raster& inputRaster,
        isce3::io::Raster& demRaster,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::product::RadarGridParameters& slicedRadarGrid,
        const isce3::product::GeoGridParameters& geoGrid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& nativeDoppler,
        const isce3::core::LUT2d<double>& imageGridDoppler,
        const isce3::core::Ellipsoid& ellipsoid, const double& thresholdGeo2rdr,
        const int& numiterGeo2rdr, const size_t& linesPerBlock,
        const bool flatten, const bool reramp,
        const AzRgFunc& azCarrierPhase, const AzRgFunc& rgCarrierPhase,
        const isce3::core::LUT2d<double>& azTimeCorrection,
        const isce3::core::LUT2d<double>& sRangeCorrection,
        const bool flattenWithCorrectedSRng,
        const std::complex<float> invalidValue,
        isce3::io::Raster* carrierPhaseRaster,
        isce3::io::Raster* flattenPhaseRaster)
{
    validate_slice(radarGrid, slicedRadarGrid);

    pyre::journal::debug_t debug("isce.geocode.geocodeSlc.geocodeSlc");

    // number of bands in the input raster
    size_t nbands = inputRaster.numBands();
    debug << "nbands: " << nbands << pyre::journal::endl;
    // create projection based on _epsg code
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(geoGrid.epsg()));

    // Interpolator pointer
    auto sincInterp = std::make_unique<
            isce3::core::Sinc2dInterpolator<std::complex<float>>>(
            isce3::core::SINC_LEN, isce3::core::SINC_SUB);

    // Compute number of blocks in the output geocoded grid
    size_t nBlocks = (geoGrid.length() + linesPerBlock - 1) / linesPerBlock;

    debug << "nBlocks: " << nBlocks << pyre::journal::endl;
    // loop over the blocks of the geocoded Grid
    for (size_t block = 0; block < nBlocks; ++block) {
        debug << "block: " << block << pyre::journal::endl;
        // Get block extents (of the geocoded grid)
        size_t lineStart, geoBlockLength;
        lineStart = block * linesPerBlock;
        if (block == (nBlocks - 1)) {
            geoBlockLength = geoGrid.length() - lineStart;
        } else {
            geoBlockLength = linesPerBlock;
        }

        // get a DEM interpolator for a block of DEM for the current geocoded
        // grid
        isce3::geometry::DEMInterpolator demInterp =
            isce3::geometry::DEMRasterToInterpolator(demRaster, geoGrid,
                    lineStart, geoBlockLength, geoGrid.width());

        // X and Y indices (in the radar coordinates) for the geocoded pixels
        // (after geo2rdr computation) - initialized to invalid values
        isce3::core::Matrix<double> rangeIndices(geoBlockLength, geoGrid.width());
        rangeIndices.fill(std::real(invalidValue));

        isce3::core::Matrix<double> azimuthIndices(geoBlockLength, geoGrid.width());
        azimuthIndices.fill(std::real(invalidValue));

        // selectively use uncorrected slant range - initialized to invalid
        // values
        isce3::core::Matrix<double> uncorrectedSRange;
        if (!flattenWithCorrectedSRng) {
            uncorrectedSRange.resize(geoBlockLength, geoGrid.width());
            uncorrectedSRange.fill(std::real(invalidValue));
        }

        // First and last line of the data block in radar coordinates
        int azimuthFirstLine = radarGrid.length() - 1;
        int azimuthLastLine = 0;

        // First and last pixel of the data block in radar coordinates
        int rangeFirstPixel = radarGrid.width() - 1;
        int rangeLastPixel = 0;

        // Compute radar coordinates of each geocoded pixel
        // Determine boundary of corresponding radar raster
        size_t geoGridWidth = geoGrid.width();
// Loop over lines, samples of the output grid
#pragma omp parallel for reduction(min                                    \
                                   : azimuthFirstLine,                    \
                                     rangeFirstPixel)                     \
        reduction(max                                                     \
                  : azimuthLastLine, rangeLastPixel)
        for (size_t blockLine = 0; blockLine < geoBlockLength; ++blockLine) {
            // Global line index
            const size_t line = lineStart + blockLine;

            for (size_t pixel = 0; pixel < geoGridWidth; ++pixel) {
                // y coordinate in the out put grid
                // Assuming geoGrid.startY() and geoGrid.startX() represent the top-left
                // corner of the first pixel, then 0.5 pixel shift is needed to get
                // to the center of each pixel
                double y = geoGrid.startY() + geoGrid.spacingY() * (line + 0.5);

                // x in the output geocoded Grid
                double x = geoGrid.startX() + geoGrid.spacingX() * (pixel + 0.5);

                // compute the azimuth time and slant range for the
                // x,y coordinates in the output grid
                double aztime, srange;
                aztime = radarGrid.sensingMid();

                // coordinate in the output projection system
                const isce3::core::Vec3 xyz {x, y, 0.0};

                // transform the xyz in the output projection system to llh
                isce3::core::Vec3 llh = proj->inverse(xyz);

                // interpolate the height from the DEM for this pixel
                llh[2] = demInterp.interpolateLonLat(llh[0], llh[1]);

                // Perform geo->rdr iterations
                int geostat = isce3::geometry::geo2rdr(
                        llh, ellipsoid, orbit, imageGridDoppler, aztime, srange,
                        radarGrid.wavelength(), radarGrid.lookSide(),
                        thresholdGeo2rdr, numiterGeo2rdr, 1.0e-8);

                // Check convergence
                if (geostat == 0)
                    continue;

                // save uncorrected slant range
                if (!flattenWithCorrectedSRng)
                    uncorrectedSRange(blockLine, pixel) = srange;

                // apply timing corrections
                // if default LUT2d used for both azimuth time and slant range
                // corrections, uncorrected slant range and corrected slant
                // range will be the same
                if (azTimeCorrection.contains(aztime, srange)) {
                    const auto aztimeCor = azTimeCorrection.eval(aztime,
                                                                 srange);
                    aztime += aztimeCor;
                }

                if (sRangeCorrection.contains(aztime, srange)) {
                    const auto srangeCor = sRangeCorrection.eval(aztime,
                                                                 srange);
                    srange += srangeCor;
                }

                // check if az time and slant within radar grid
                if (!slicedRadarGrid.contains(aztime, srange)
                        || !nativeDoppler.contains(aztime, srange))
                    continue;

                // get the row and column index in the radar grid
                double azimuthCoord = (aztime - radarGrid.sensingStart()) * radarGrid.prf();
                double rangeCoord = (srange - radarGrid.startingRange()) /
                              radarGrid.rangePixelSpacing();

                azimuthFirstLine = std::min(
                        azimuthFirstLine, static_cast<int>(std::floor(azimuthCoord)));
                azimuthLastLine =
                        std::max(azimuthLastLine,
                                 static_cast<int>(std::ceil(azimuthCoord) - 1));
                rangeFirstPixel = std::min(rangeFirstPixel,
                                           static_cast<int>(std::floor(rangeCoord)));
                rangeLastPixel = std::max(
                        rangeLastPixel, static_cast<int>(std::ceil(rangeCoord) - 1));

                // store the adjusted X and Y indices
                rangeIndices(blockLine, pixel) = rangeCoord;
                azimuthIndices(blockLine, pixel) = azimuthCoord;
            }
        } // end loops over lines and pixel of output grid

        // Fill the output block with the default value before checking validity
        isce3::core::EArray2D<std::complex<float>> geoDataBlock(geoBlockLength,
                                                                geoGrid.width());

        // assume all values invalid by default
        // interpolate and carrierPhaseRerampAndFlatten will only modify valid pixels
        geoDataBlock.fill(invalidValue);

        // init phase and range offset blocks, but only resize and fill if
        // their respective raster pointers are not nullptr
        isce3::core::EArray2D<double> carrierPhaseBlock;
        if (carrierPhaseRaster) {
            carrierPhaseBlock.resize(geoBlockLength, geoGrid.width());
            carrierPhaseBlock.fill(invalidValue.real());
        }

        isce3::core::EArray2D<double> flattenPhaseBlock;
        if (flattenPhaseRaster) {
            flattenPhaseBlock.resize(geoBlockLength, geoGrid.width());
            flattenPhaseBlock.fill(invalidValue.real());
        }

        // Extra margin for interpolation to avoid gaps between blocks in output
        int interp_margin = 5;

        // Get min and max swath extents from among all threads
        azimuthFirstLine = std::max(azimuthFirstLine - interp_margin, 0);
        rangeFirstPixel = std::max(rangeFirstPixel - interp_margin, 0);

        azimuthLastLine = std::min(azimuthLastLine + interp_margin,
                                   static_cast<int>(radarGrid.length() - 1));
        rangeLastPixel = std::min(rangeLastPixel + interp_margin,
                                  static_cast<int>(radarGrid.width() - 1));

        if (azimuthFirstLine > azimuthLastLine ||
            rangeFirstPixel > rangeLastPixel) {
            // No valid pixels in this block, so set to invalid and continue
            for (size_t band = 0; band < nbands; ++band) {
                outputRaster.setBlock(geoDataBlock.data(), 0, lineStart,
                                    geoGrid.width(), geoBlockLength, band + 1);
            }
            // set output if phase and range offset rasters not nullptr
            if (carrierPhaseRaster) {
                carrierPhaseRaster->setBlock(carrierPhaseBlock.data(), 0, lineStart,
                        geoGrid.width(), geoBlockLength, 1);
            }

            if (flattenPhaseRaster) {
                flattenPhaseRaster->setBlock(flattenPhaseBlock.data(), 0, lineStart,
                        geoGrid.width(), geoBlockLength, 1);
            }
            continue;
        }

        // shape of the required block of data in the radar coordinates
        size_t rdrBlockLength = azimuthLastLine - azimuthFirstLine + 1;
        size_t rdrBlockWidth = rangeLastPixel - rangeFirstPixel + 1;

        // define the matrix based on the rasterbands data type
        isce3::core::EArray2D<std::complex<float>> rdrDataBlock(rdrBlockLength,
                                                                rdrBlockWidth);

        // fill both radar data block with zero
        rdrDataBlock.fill(0);

        // for each band in the input:
        for (size_t band = 0; band < nbands; ++band) {

            debug << "band: " << band << pyre::journal::newline;
            // get a block of data
            debug << "get data block " << pyre::journal::endl;
            inputRaster.getBlock(rdrDataBlock.data(), rangeFirstPixel,
                                 azimuthFirstLine, rdrBlockWidth,
                                 rdrBlockLength, band + 1);

            // Remove doppler and carriers as needd
            carrierPhaseDeramp(rdrDataBlock, azCarrierPhase, rgCarrierPhase,
                   azimuthFirstLine, rangeFirstPixel, radarGrid);

            // interpolate the data in radar grid to the geocoded grid.
            interpolate(rdrDataBlock, geoDataBlock, rangeIndices, azimuthIndices,
                    azimuthFirstLine, rangeFirstPixel, sincInterp.get(),
                    radarGrid, nativeDoppler);

            // Add back doppler and carriers as needd
            carrierPhaseRerampAndFlatten(geoDataBlock, carrierPhaseBlock,
                    flattenPhaseBlock, rdrDataBlock, azCarrierPhase,
                    rgCarrierPhase, nativeDoppler, rangeIndices,
                    azimuthIndices, radarGrid, flatten, reramp,
                    azimuthFirstLine, rangeFirstPixel, flattenWithCorrectedSRng,
                    uncorrectedSRange);

            // set output
            debug << "set output " << pyre::journal::endl;
            outputRaster.setBlock(geoDataBlock.data(), 0, lineStart,
                                  geoGrid.width(), geoBlockLength, band + 1);

        }
        // set output if phase and range offset rasters not nullptr
        if (carrierPhaseRaster) {
            carrierPhaseRaster->setBlock(carrierPhaseBlock.data(), 0,
                    lineStart, geoGrid.width(), geoBlockLength, 1);
        }

        if (flattenPhaseRaster) {
            flattenPhaseRaster->setBlock(flattenPhaseBlock.data(), 0,
                    lineStart, geoGrid.width(), geoBlockLength, 1);
        }
        // set output block of data
    } // end loop over block of output grid
}


template<typename AzRgFunc>
void geocodeSlc(
        std::vector<EArray2dc64>& geoDataBlocks,
        EArray2df64 carrierPhaseBlock,
        EArray2df64 flattenPhaseBlock,
        const std::vector<EArray2dc64>& rdrDataBlocks,
        isce3::io::Raster& demRaster,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::product::RadarGridParameters& slicedRadarGrid,
        const isce3::product::GeoGridParameters& geoGrid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& nativeDoppler,
        const isce3::core::LUT2d<double>& imageGridDoppler,
        const isce3::core::Ellipsoid& ellipsoid,
        const double& thresholdGeo2rdr, const int& numiterGeo2rdr,
        const size_t& azimuthFirstLine, const size_t& rangeFirstPixel,
        const bool flatten, const bool reramp,
        const AzRgFunc& azCarrierPhase,
        const AzRgFunc& rgCarrierPhase,
        const isce3::core::LUT2d<double>& azTimeCorrection,
        const isce3::core::LUT2d<double>& sRangeCorrection,
        const bool flattenWithCorrectedSRng,
        const std::complex<float> invalidValue)
{
    if (geoDataBlocks.size() != rdrDataBlocks.size()) {
        std::string error_msg("number of geoDataBlocks != number of rdrDataBlocks");
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    // check all input and output arrays are of the same size
    if (!array_sizes_consistent(geoDataBlocks)) {
        std::string error_msg("geoDataBlocks arrays do not have the same size");
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
    if (!array_sizes_consistent(rdrDataBlocks)) {
        std::string error_msg("rdrDataBlocks arrays do not have the same size");
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    for (auto geoDataBlock : geoDataBlocks)
        geoDataBlock.fill(invalidValue);

    validate_slice(radarGrid, slicedRadarGrid);

    // create projection based on _epsg code
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(geoGrid.epsg()));

    // Interpolator pointer
    auto sincInterp = std::make_unique<
            isce3::core::Sinc2dInterpolator<std::complex<float>>>(
            isce3::core::SINC_LEN, isce3::core::SINC_SUB);

    // get a DEM interpolator for a block of DEM for the current geocoded
    // grid
    isce3::geometry::DEMInterpolator demInterp =
        isce3::geometry::DEMRasterToInterpolator(demRaster, geoGrid, 0,
                geoGrid.length(), geoGrid.width());

    // X and Y indices (in the radar coordinates) for the geocoded pixels
    // (after geo2rdr computation)
    isce3::core::Matrix<double> rangeIndices(geoGrid.length(), geoGrid.width());
    isce3::core::Matrix<double> azimuthIndices(geoGrid.length(), geoGrid.width());

    // fill with invalid value
    rangeIndices.fill(std::real(invalidValue));
    azimuthIndices.fill(std::real(invalidValue));

    // Resize and fill uncorrected slant range if flag set
    isce3::core::Matrix<double> uncorrectedSRange;
    if (!flattenWithCorrectedSRng) {
        uncorrectedSRange.resize(geoGrid.length(), geoGrid.width());
        uncorrectedSRange.fill(std::real(invalidValue));
    }

    // Compute radar coordinates of each geocoded pixel
    // Determine boundary of corresponding radar raster
    size_t geoGridWidth = geoGrid.width();
// Loop over lines, samples of the output grid
#pragma omp parallel for
    for (size_t line = 0; line < geoGrid.length(); ++line) {
        for (size_t pixel = 0; pixel < geoGridWidth; ++pixel) {
            // y coordinate in the out put grid
            // Assuming geoGrid.startY() and geoGrid.startX() represent the top-left
            // corner of the first pixel, then 0.5 pixel shift is needed to get
            // to the center of each pixel
            double y = geoGrid.startY() + geoGrid.spacingY() * (line + 0.5);

            // x in the output geocoded Grid
            double x = geoGrid.startX() + geoGrid.spacingX() * (pixel + 0.5);

            // compute the azimuth time and slant range for the
            // x,y coordinates in the output grid
            double aztime, srange;
            aztime = radarGrid.sensingMid();

            // coordinate in the output projection system
            const isce3::core::Vec3 xyz {x, y, 0.0};

            // transform the xyz in the output projection system to llh
            isce3::core::Vec3 llh = proj->inverse(xyz);

            // interpolate the height from the DEM for this pixel
            llh[2] = demInterp.interpolateLonLat(llh[0], llh[1]);

            // Perform geo->rdr iterations
            int geostat = isce3::geometry::geo2rdr(
                    llh, ellipsoid, orbit, imageGridDoppler, aztime, srange,
                    radarGrid.wavelength(), radarGrid.lookSide(),
                    thresholdGeo2rdr, numiterGeo2rdr, 1.0e-8);

            // Check convergence
            if (geostat == 0) {
                continue;
            }

            // save uncorrected slant range
            if (!flattenWithCorrectedSRng)
                uncorrectedSRange(line, pixel) = srange;

            // apply timing corrections
            // if default LUT2d used for both azimuth time and slant range
            // corrections, uncorrected slant range and corrected slant
            // range will be the same
            if (azTimeCorrection.contains(aztime, srange)) {
                const auto aztimeCor = azTimeCorrection.eval(aztime,
                                                             srange);
                aztime += aztimeCor;
            }

            if (sRangeCorrection.contains(aztime, srange)) {
                const auto srangeCor = sRangeCorrection.eval(aztime,
                                                             srange);
                srange += srangeCor;
            }

            // get the row and column index in the radar grid
            double azimuthCoord = (aztime - radarGrid.sensingStart()) * radarGrid.prf();
            double rangeCoord = (srange - radarGrid.startingRange()) /
                          radarGrid.rangePixelSpacing();

            if (!slicedRadarGrid.contains(aztime, srange)
                    || !nativeDoppler.contains(aztime, srange))
                continue;

            // store the adjusted X and Y indices
            rangeIndices(line, pixel) = rangeCoord;
            azimuthIndices(line, pixel) = azimuthCoord;
        }
    } // end loops over lines and pixel of output grid

    // loop over pairs of radar and geo block array
    for (auto [gIt, rIt] = std::tuple(geoDataBlocks.begin(), rdrDataBlocks.begin());
            gIt != geoDataBlocks.end(); ++gIt, ++rIt) {
        auto geoDataBlock = *gIt;
        auto rdrDataBlock = *rIt;

        // interpolate and carrierPhaseRerampAndFlatten will only modify valid pixels
        // Remove doppler and carriers as needd
        carrierPhaseDeramp(rdrDataBlock, azCarrierPhase, rgCarrierPhase,
                azimuthFirstLine, rangeFirstPixel, radarGrid);

        // interpolate the data in radar grid to the geocoded grid.
        interpolate(rdrDataBlock, geoDataBlock, rangeIndices, azimuthIndices,
                azimuthFirstLine, rangeFirstPixel, sincInterp.get(),
                radarGrid, nativeDoppler);

        // Add back doppler and carriers as needd
        carrierPhaseRerampAndFlatten(geoDataBlock, carrierPhaseBlock,
                flattenPhaseBlock, rdrDataBlock, azCarrierPhase,
                rgCarrierPhase, nativeDoppler, rangeIndices,
                azimuthIndices, radarGrid, flatten, reramp,
                azimuthFirstLine, rangeFirstPixel, flattenWithCorrectedSRng,
                uncorrectedSRange);
    }
}

#define EXPLICIT_INSTANTIATION(AzRgFunc)                                \
template void geocodeSlc<AzRgFunc>(                                     \
        isce3::io::Raster& outputRaster, isce3::io::Raster& inputRaster,\
        isce3::io::Raster& demRaster,                                   \
        const isce3::product::RadarGridParameters& radarGrid,           \
        const isce3::product::GeoGridParameters& geoGrid,               \
        const isce3::core::Orbit& orbit,                                \
        const isce3::core::LUT2d<double>& nativeDoppler,                \
        const isce3::core::LUT2d<double>& imageGridDoppler,             \
        const isce3::core::Ellipsoid& ellipsoid,                        \
        const double& thresholdGeo2rdr,                                 \
        const int& numiterGeo2rdr, const size_t& linesPerBlock,         \
        const bool flatten,  const bool reramp,                         \
        const AzRgFunc& azCarrierPhase, const AzRgFunc& rgCarrierPhase, \
        const isce3::core::LUT2d<double>& azTimeCorrection,             \
        const isce3::core::LUT2d<double>& sRangeCorrection,             \
        const bool flattenWithCorrectedSRng,                            \
        const std::complex<float> invalidValue,                         \
        isce3::io::Raster* phaseRaster,                                 \
        isce3::io::Raster* rgOffsetRaster);                             \
template void geocodeSlc<AzRgFunc>(                                     \
        isce3::io::Raster& outputRaster, isce3::io::Raster& inputRaster,\
        isce3::io::Raster& demRaster,                                   \
        const isce3::product::RadarGridParameters& radarGrid,           \
        const isce3::product::RadarGridParameters& slicedRadarGrid,     \
        const isce3::product::GeoGridParameters& geoGrid,               \
        const isce3::core::Orbit& orbit,                                \
        const isce3::core::LUT2d<double>& nativeDoppler,                \
        const isce3::core::LUT2d<double>& imageGridDoppler,             \
        const isce3::core::Ellipsoid& ellipsoid,                        \
        const double& thresholdGeo2rdr,                                 \
        const int& numiterGeo2rdr, const size_t& linesPerBlock,         \
        const bool flatten,  const bool reramp,                         \
        const AzRgFunc& azCarrierPhase, const AzRgFunc& rgCarrierPhase, \
        const isce3::core::LUT2d<double>& azTimeCorrection,             \
        const isce3::core::LUT2d<double>& sRangeCorrection,             \
        const bool flattenWithCorrectedSRng,                            \
        const std::complex<float> invalidValue,                         \
        isce3::io::Raster* phaseRaster,                                 \
        isce3::io::Raster* rgOffsetRaster);                             \
template void geocodeSlc<AzRgFunc>(                                     \
        std::vector<EArray2dc64>& geoDataBlocks,                        \
        EArray2df64 carrierPhaseBlock,                                  \
        EArray2df64 flattenPhaseBlock,                                  \
        const std::vector<EArray2dc64>& rdrDataBlocks,                  \
        isce3::io::Raster& demRaster,                                   \
        const isce3::product::RadarGridParameters& radarGrid,           \
        const isce3::product::RadarGridParameters& slicedRadarGrid,     \
        const isce3::product::GeoGridParameters& geoGrid,               \
        const isce3::core::Orbit& orbit,                                \
        const isce3::core::LUT2d<double>& nativeDoppler,                \
        const isce3::core::LUT2d<double>& imageGridDoppler,             \
        const isce3::core::Ellipsoid& ellipsoid,                        \
        const double& thresholdGeo2rdr, const int& numiterGeo2rdr,      \
        const size_t& azimuthFirstLine, const size_t& rangeFirstPixel,  \
        const bool flatten,  const bool reramp,                         \
        const AzRgFunc& azCarrierPhase, const AzRgFunc& rgCarrierPhase, \
        const isce3::core::LUT2d<double>& azTimeCorrection,             \
        const isce3::core::LUT2d<double>& sRangeCorrection,             \
        const bool flattenWithCorrectedSRng,                            \
        const std::complex<float> invalidValue                         \
        )

EXPLICIT_INSTANTIATION(isce3::core::LUT2d<double>);
EXPLICIT_INSTANTIATION(isce3::core::Poly2d);

} // namespace isce3::geocode
