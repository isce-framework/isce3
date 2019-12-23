#pragma once

#include <gdal_priv.h>
#include <string>

#include "Dataset.h"

namespace isce { namespace io { namespace gdal {

/** Wrapper for GDALRasterBand representing a single raster */
class Raster {
public:

    /** Default GDAL driver for raster creation */
    static
    std::string defaultDriver() { return Dataset::defaultDriver(); }

    /**
     * Open an existing file containing a single raster band as a GDAL raster.
     *
     * \param[in] path File path
     * \param[in] access Access mode
     */
    Raster(const std::string & path, GDALAccess access = GA_ReadOnly);

    /**
     * Open an existing file as a GDAL dataset and fetch the specified raster band.
     *
     * \param[in] path File path
     * \param[in] band Raster band index (1-based)
     * \param[in] access Access mode
     */
    Raster(const std::string & path, int band, GDALAccess access = GA_ReadOnly);

    /**
     * Create a new GDAL dataset containing a single raster band.
     *
     * \param[in] path File path
     * \param[in] width Number of columns
     * \param[in] length Number of rows
     * \param[in] datatype Data type identifier
     * \param[in] driver GDAL driver name
     */
    Raster(const std::string & path,
           int width,
           int length,
           GDALDataType datatype,
           const std::string & driver = defaultDriver());

    /**
     * Create a read-only raster describing an existing in-memory array.
     *
     * The data layout is assumed to be in row major format.
     *
     * \param[in] data Pointer to first pixel
     * \param[in] width Number of columns
     * \param[in] length Number of rows
     */
    template<typename T>
    Raster(const T * data, int width, int length);

    /**
     * Create a raster describing an existing in-memory array.
     *
     * The data layout is assumed to be in row major format.
     *
     * \param[in] data Pointer to first pixel
     * \param[in] width Number of columns
     * \param[in] length Number of rows
     * \param[in] access Access mode
     */
    template<typename T>
    Raster(T * data, int width, int length, GDALAccess access = GA_Update);

    /**
     * Create a read-only raster describing an existing in-memory array.
     *
     * \param[in] data Pointer to first pixel
     * \param[in] width Number of columns
     * \param[in] length Number of rows
     * \param[in] colstride Stride in bytes between the start of adjacent columns
     * \param[in] rowstride Stride in bytes between the start of adjacent rows
     */
    template<typename T>
    Raster(const T * data,
           int width,
           int length,
           std::size_t colstride,
           std::size_t rowstride);

    /**
     * Create a raster describing an existing in-memory array.
     *
     * \param[in] data Pointer to first pixel
     * \param[in] width Number of columns
     * \param[in] length Number of rows
     * \param[in] colstride Stride in bytes between the start of adjacent columns
     * \param[in] rowstride Stride in bytes between the start of adjacent rows
     * \param[in] access Access mode
     */
    template<typename T>
    Raster(T * data,
           int width,
           int length,
           std::size_t colstride,
           std::size_t rowstride,
           GDALAccess access = GA_Update);

    /** Get the dataset containing the raster */
    const Dataset & dataset() const { return _dataset; }

    /** Band index (1-based) */
    int band() const { return _band; }

    /** Datatype identifier */
    GDALDataType datatype() const;

    /** Access mode */
    GDALAccess access() const { return _dataset.access(); }

    /** Number of columns */
    int width() const { return _dataset.width(); }

    /** Number of rows */
    int length() const { return _dataset.length(); }

    /** Driver name */
    std::string driver() const { return _dataset.driver(); }

    /**
     * Get transform from raster coordinates (pixel, line) to projected
     * coordinates (x, y)
     *
     * If no transform is found, the default (identity) transform is used.
     */
    GeoTransform getGeoTransform() const { return _dataset.getGeoTransform(); }

    /**
     * Set geotransform
     *
     * \throws isce::except::GDALError if the format does not support this operation
     */
    void setGeoTransform(const GeoTransform & transform) { _dataset.setGeoTransform(transform); }

    /**
     * Get spatial reference system
     *
     * \throws isce::except::GDALError if the spatial reference system is unavailable
     */
    isce::core::ProjectionBase * getProjection() const { return _dataset.getProjection(); }

    /**
     * Set spatial reference system
     *
     * \throws isce::except::GDALError if the format does not support this operation
     */
    void setProjection(const isce::core::ProjectionBase * proj) { _dataset.setProjection(proj); }

    /** Left edge of left-most pixel in projection coordinates */
    double x0() const { return _dataset.x0(); }

    /** Upper edge of upper-most line in projection coordinates */
    double y0() const { return _dataset.y0(); }

    /** Pixel width in projection coordinates */
    double dx() const { return _dataset.dx(); }

    /** Line height in projection coordinates */
    double dy() const { return _dataset.dy(); }

    /**
     * Read a single pixel value from the raster.
     *
     * The data will be converted to the requested type if different than the
     * native datatype. If \p T is void, no conversion is performed.
     *
     * \param[out] dst Destination buffer
     * \param[in] col Column index
     * \param[in] row Row index
     */
    template<typename T>
    void readPixel(T * dst, int col, int row) const;

    /**
     * Write a single pixel value to the raster.
     *
     * The data will be converted from the input type if different than the
     * native datatype. If \p T is void, no conversion is performed.
     *
     * \param[in] src Source value
     * \param[in] col Column index
     * \param[in] row Row index
     */
    template<typename T>
    void writePixel(const T * src, int col, int row);

    /**
     * Read a line of pixel data from the raster.
     *
     * The data will be converted to the requested type if different than the
     * native datatype. If \p T is void, no conversion is performed.
     *
     * \param[out] dst Destination buffer
     * \param[in] row Row index
     */
    template<typename T>
    void readLine(T * dst, int row) const;

    /**
     * Write a line of pixel data to the raster.
     *
     * The data will be converted from the input type if different than the
     * native datatype. If \p T is void, no conversion is performed.
     *
     * \param[in] src Source values
     * \param[in] row Row index
     */
    template<typename T>
    void writeLine(const T * src, int row);

    /**
     * Read one or more lines of pixel data from the raster.
     *
     * The data will be converted to the requested type if different than the
     * native datatype. If \p T is void, no conversion is performed.
     *
     * \param[out] dst Destination buffer
     * \param[in] first_row Index of first row
     * \param[in] num_rows Number of rows
     */
    template<typename T>
    void readLines(T * dst, int first_row, int num_rows) const;

    /**
     * Write one or more lines of pixel data to the raster.
     *
     * The data will be converted from the input type if different than the
     * native datatype. If \p T is void, no conversion is performed.
     *
     * \param[in] src Source values
     * \param[in] first_row Index of first row
     * \param[in] num_rows Number of rows
     */
    template<typename T>
    void writeLines(const T * src, int first_row, int num_rows);

    /**
     * Read a block of pixel data from the raster.
     *
     * The data will be converted to the requested type if different than the
     * native datatype. If \p T is void, no conversion is performed.
     *
     * \param[out] dst Destination buffer
     * \param[in] first_col Index of first column
     * \param[in] first_row Index of first row
     * \param[in] num_cols Number of columns
     * \param[in] num_rows Number of rows
     */
    template<typename T>
    void readBlock(T * dst, int first_col, int first_row, int num_cols, int num_rows) const;

    /**
     * Write a block of pixel data to the raster.
     *
     * The data will be converted from the input type if different than the
     * native datatype. If \p T is void, no conversion is performed.
     *
     * \param[in] src Source values
     * \param[in] first_col Index of first column
     * \param[in] first_row Index of first row
     * \param[in] num_cols Number of columns
     * \param[in] num_rows Number of rows
     */
    template<typename T>
    void writeBlock(const T * src, int first_col, int first_row, int num_cols, int num_rows);

    /**
     * Read all pixel data from the raster.
     *
     * The data will be converted to the requested type if different than the
     * native datatype. If \p T is void, no conversion is performed.
     *
     * \param[out] dst Destination buffer
     */
    template<typename T>
    void readAll(T * dst) const;

    /**
     * Write all pixel data to the raster.
     *
     * The data will be converted from the input type if different than the
     * native datatype. If \p T is void, no conversion is performed.
     *
     * \param[in] src Source values
     */
    template<typename T>
    void writeAll(const T * src);

    /** Get the underlying GDALRasterBand pointer */
    GDALRasterBand * get() { return _dataset._dataset->GetRasterBand(_band); }

    /** Get the underlying GDALRasterBand pointer */
    const GDALRasterBand * get() const { return _dataset._dataset->GetRasterBand(_band); }

    friend class Dataset;

private:

    Raster(const Dataset & dataset, int band);

    template<typename T>
    GDALDataType getIODataType() const;

    template<typename T>
    CPLErr readwriteBlock(T * buf,
                          int first_col,
                          int first_row,
                          int num_cols,
                          int num_rows,
                          GDALRWFlag rwflag) const;

    Dataset _dataset;
    int _band = 1;
};

}}}

#define ISCE_IO_GDAL_RASTER_ICC
#include "Raster.icc"
#undef ISCE_IO_GDAL_RASTER_ICC
