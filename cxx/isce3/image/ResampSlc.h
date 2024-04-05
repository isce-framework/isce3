#pragma once

#include "forward.h"
#include <isce3/io/forward.h>

#include <complex>
#include <cstdint>
#include <cstdio>
#include <valarray>

#include <pyre/journal.h>

#include <isce3/core/Interpolator.h>
#include <isce3/core/Poly2d.h>
#include <isce3/product/RadarGridProduct.h>
#include <isce3/product/RadarGridParameters.h>

namespace isce3 { namespace image {

class ResampSlc {
public:
    typedef Tile<std::complex<float>> Tile_t;

    /** Constructor from an isce3::product::RadarGridProduct (no flattening) */
    ResampSlc(const isce3::product::RadarGridProduct& product, char frequency = 'A',
              const std::complex<float> invalid_value = std::complex<float>(0.0));

    /**
     * Constructor from an isce3::product::RadarGridProduct and reference product
     * (flattening)
     */
    ResampSlc(const isce3::product::RadarGridProduct& product,
              const isce3::product::RadarGridProduct& refProduct, char frequency = 'A',
              const std::complex<float> invalid_value = std::complex<float>(0.0));

    /** Constructor from an isce3::product::Swath (no flattening) */
    ResampSlc(const isce3::product::Swath& swath,
              const std::complex<float> invalid_value = std::complex<float>(0.0));

    /**
     * Constructor from an isce3::product::Swath and reference swath
     * (flattening)
     */
    ResampSlc(const isce3::product::Swath& swath,
              const isce3::product::Swath& refSwath,
              const std::complex<float> invalid_value = std::complex<float>(0.0));

    /**
     * Constructor from an isce3::product::RadarGridParameters (no flattening)
     */
    ResampSlc(const isce3::product::RadarGridParameters& rdr_grid,
              const isce3::core::LUT2d<double>& doppler,
              const std::complex<float> invalid_value = std::complex<float>(0.0));

    /**
     * Constructor from an isce3::product::RadarGridParameters and reference
     * radar grid (flattening)
     */
    ResampSlc(const isce3::product::RadarGridParameters& rdr_grid,
              const isce3::product::RadarGridParameters& ref_rdr_grid,
              const isce3::core::LUT2d<double>& doppler,
              const std::complex<float> invalid_value = std::complex<float>(0.0));

    /** Constructor from individual components (no flattening) */
    ResampSlc(const isce3::core::LUT2d<double>& doppler, double startingRange,
              double rangePixelSpacing, double sensingStart, double prf,
              double wvl,
              const std::complex<float> invalid_value = std::complex<float>(0.0));

    /** Constructor from individual components (flattening) */
    ResampSlc(const isce3::core::LUT2d<double>& doppler, double startingRange,
              double rangePixelSpacing, double sensingStart, double prf,
              double wvl, double refStartingRange, double refRangePixelSpacing,
              double refWvl,
              const std::complex<float> invalid_value = std::complex<float>(0.0));

    /** Destructor */
    ~ResampSlc() {};

    double startingRange() const { return _startingRange; }
    double rangePixelSpacing() const { return _rangePixelSpacing; }
    double sensingStart() const { return _sensingStart; }
    double prf() const { return _prf; }
    double wavelength() const { return _wavelength; }
    double refStartingRange() const { return _refStartingRange; }
    double refRangePixelSpacing() const { return _refRangePixelSpacing; }
    double refWavelength() const { return _refWavelength; }

    // Poly2d and LUT getters
    isce3::core::Poly2d rgCarrier() const;
    isce3::core::Poly2d azCarrier() const;

    // Poly2d and LUT setters
    void rgCarrier(const isce3::core::Poly2d&);
    void azCarrier(const isce3::core::Poly2d&);

    /** Get read-only reference to Doppler LUT2d */
    const isce3::core::LUT2d<double>& doppler() const;

    /** Get reference to Doppler LUT2d */
    isce3::core::LUT2d<double>& doppler();

    /** Set Doppler LUT2d */
    void doppler(const isce3::core::LUT2d<double>&);

    // Set reference product for flattening
    void referenceProduct(const isce3::product::RadarGridProduct& product,
                          char frequency = 'A');

    // Get/set number of lines per processing tile
    size_t linesPerTile() const;
    void linesPerTile(size_t);

    /** Get flag for reference data */
    bool haveRefData() const { return _haveRefData; }

    // Convenience functions
    void declare(size_t, size_t, size_t, size_t) const;

    /* Generic resamp entry point from externally created rasters
     *
     * \param[in] inputSlc          input raster of SLC to be resampled
     * \param[in] outputSlc         output raster of resampled SLC
     * \param[in] rgOffsetRaster    raster of range shift to be applied
     * \param[in] azOffsetRaster    raster of azimuth shift to be applied
     * \param[in] inputBand         band of input raster to resample
     * \param[in] flatten           flag to flatten resampled SLC
     * \param[in] rowBuffer         number of rows excluded from top/bottom of azimuth
     *                              raster while searching for min/max indices of
     *                              resampled SLC
     * \param[in] chipSize          size of chip used in sinc interpolation
     */
    void resamp(isce3::io::Raster& inputSlc,
                isce3::io::Raster& outputSlc,
                isce3::io::Raster& rgOffsetRaster,
                isce3::io::Raster& azOffsetRaster,
                int inputBand = 1,
                bool flatten = false,
                int rowBuffer = 40,
                int chipSize = isce3::core::SINC_ONE);

    /* Generic resamp entry point: use filenames to create rasters
     * internally in function.
     *
     * \param[in] inputFilename     path of file containing SLC to be resampled
     * \param[in] outputFilename    path of file containing resampled SLC
     * \param[in] rgOffsetFilename  path of file containing range shift to be applied
     * \param[in] azOffsetFilename  path of file containing azimuth shift to be applied
     * \param[in] inputBand         band of input raster to resample
     * \param[in] flatten           flag to flatten resampled SLC
     * \param[in] rowBuffer         number of rows excluded from top/bottom of azimuth
     *                              raster while searching for min/max indices of
     *                              resampled SLC
     * \param[in] chipSize          size of chip used in sinc interpolation
     */
    void resamp(const std::string& inputFilename,
                const std::string& outputFilename,
                const std::string& rgOffsetFilename,
                const std::string& azOffsetFilename,
                int inputBand = 1,
                bool flatten = false,
                int rowBuffer = 40,
                int chipSize = isce3::core::SINC_ONE);

protected:
    // Number of lines per tile
    size_t _linesPerTile = 1000;
    // Band number
    int _inputBand;
    // Filename of the input product
    std::string _filename;
    // Flag indicating if we have a reference data (for flattening)
    bool _haveRefData;
    // Interpolator pointer
    isce3::core::Interpolator<std::complex<float>>* _interp;

    // Polynomials and LUTs
    isce3::core::Poly2d _rgCarrier; // range carrier polynomial
    isce3::core::Poly2d _azCarrier; // azimuth carrier polynomial
    isce3::core::LUT2d<double> _dopplerLUT;

    // Variables ingested from a Swath
    double _startingRange;
    double _rangePixelSpacing;
    double _sensingStart;
    double _prf;
    double _wavelength;
    double _refStartingRange;
    double _refRangePixelSpacing;
    double _refWavelength;

    /*
     * Initialize az/range offset tiles and load from az/range offset rasters
     * for current block
     *
     * \param[in]  tile             tile object containing input block
     *                              start/stop indices and dimensions
     * \param[in]  azOffsetRaster   raster containing azimuth offsets (from
     *                              geo2rdr)
     * \param[in]  rgOffsetRaster   raster containing range offsets (from
     *                              geo2rdr)
     * \param[out] azOffTile        tile object containing azimuth offsets for
     *                              current block
     * \param[out] rgOffTile        tile object containing range offsets for
     *                              current block
     * \param[in]  outWidth         width of az/range offset tiles
    */
    void _initializeOffsetTiles(Tile_t& tile,
                                isce3::io::Raster& azOffsetRaster,
                                isce3::io::Raster& rgOffsetRaster,
                                Tile<double>& azOffTile,
                                Tile<double>& rgOffTile,
                                size_t outWidth);

    /*
     * Initialize input SLC tile
     *
     * \param[out] til e            tile object containing input block
     *                              start/stop indices and dimensions
     * \param[out] inputSlc         raster containing input SLC used to
     *                              populate tile object
     * \param[in]  azOffTile        tile object containing azimuth offsets for
     *                              current block used to determine tile
     *                              start/stop indices and dimensions
     * \param[in]  outLength        length output for current block
     * \param[in]  rowBuffer
     * \param[in]  chipHalf         half of the size of the chip used by the
     *                              sinc interpolator
    */
    void _initializeTile(Tile_t& tile, isce3::io::Raster& inputSlc,
                         const Tile<double>& azOffTile, size_t outLength,
                         int rowBuffer, int chipHalf);


    /*
     * Transfrom input SLC tile/block by sinc interpolating
     *
     * \param[in]  tile             tile object containing input block
     *                              start/stop indices and dimensions
     * \param[in]  outputSlc        raster containing transformed/output SLC
     * \param[out] rgOffTile        tile object containing range offsets for
     *                              current block
     * \param[in]  azOffTile        tile object containing azimuth offsets for
     *                              current block
     * \param[in]  inLength         input length for current block
     * \param[in]  flatten          whether or not to flatten transformed SLC
     *                              block
     * \param[in]  chipSize         size of the chip used by the sinc
     *                              interpolator
    */
    void _transformTile(Tile_t& tile, isce3::io::Raster& outputSlc,
                        const Tile<double>& rgOffTile,
                        const Tile<double>& azOffTile, size_t inLength,
                        bool flatten, int chipSize);

    // Convenience functions
    size_t _computeNumberOfTiles(size_t, size_t);

    // Initialize interpolator pointer
    void _prepareInterpMethods(isce3::core::dataInterpMethod, int);

    // Set radar parameters from an isce3::product::Swath
    void _setDataFromSwath(const isce3::product::Swath& swath);

    // Set reference radar parameters from an isce3::product::Swath (for
    // flattening)
    void _setRefDataFromSwath(const isce3::product::Swath& swath);

    // Value assigned to invalid pixels
    // Default 0 + 0j to facilitate downstream crossmul processing
    std::complex<float> _invalid_value = std::complex<float>(0.0, 0.0);
};

}} // namespace isce3::image

// Get inline implementations for ResampSlc
#include "ResampSlc.icc"
