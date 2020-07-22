#pragma once

#include "forward.h"

#include <gdal_priv.h>
#include <memory>
#include <string>

#include <isce3/core/Projections.h>
#include <isce3/io/IH5.h>

#include "GeoTransform.h"

namespace isce3 { namespace io { namespace gdal {

/** Wrapper for GDALDataset representing a collection of associated Raster bands */
class Dataset {
public:

    /** Default GDAL driver for dataset creation */
    static
    std::string defaultDriver() { return "ENVI"; }

    /**
     * Open an existing file as a GDAL dataset.
     *
     * \param[in] path File path
     * \param[in] access Access mode
     */
    Dataset(const std::string & path, GDALAccess access = GA_ReadOnly);

    /**
     * Create a dataset from an HDF5 dataset
     *
     * The resulting dataset is invalidated if the HDF5 file is closed.
     *
     * \param[in] dataset   HDF5 dataset
     * \param[in] access    Access mode
     */
    Dataset(const isce3::io::IDataSet & dataset, GDALAccess access = GA_ReadOnly);

    /**
     * Create a new GDAL dataset.
     *
     * \param[in] path File path
     * \param[in] width Number of columns
     * \param[in] length Number of rows
     * \param[in] bands Number of bands
     * \param[in] datatype Data type identifier
     * \param[in] driver GDAL driver name
     */
    Dataset(const std::string & path,
            int width,
            int length,
            int bands,
            GDALDataType datatype,
            const std::string & driver = defaultDriver());

    /**
     * Create a new GDAL dataset as a copy of an existing dataset.
     *
     * The duplicate is created with the same driver format as the original.
     *
     * \param[in] path File path of new dataset
     * \param[in] src Source dataset
     */
    Dataset(const std::string & path, const Dataset & src);

    /**
     * Create a new GDAL dataset as a copy of an existing dataset.
     *
     * \param[in] path File path of new dataset
     * \param[in] src Source dataset
     * \param[in] driver GDAL driver name
     */
    Dataset(const std::string & path, const Dataset & src, const std::string & driver);

    /**
     * Create a read-only dataset describing an existing in-memory array.
     *
     * The data layout is assumed to be in row major, band sequential format.
     *
     * \param[in] data Pointer to first pixel of first band
     * \param[in] width Number of columns
     * \param[in] length Number of rows
     * \param[in] bands Number of bands
     */
    template<typename T>
    Dataset(const T * data, int width, int length, int bands);

    /**
     * Create a dataset describing an existing in-memory array.
     *
     * The data layout is assumed to be in row major, band sequential format.
     *
     * \param[in] data Pointer to first pixel of first band
     * \param[in] width Number of columns
     * \param[in] length Number of rows
     * \param[in] bands Number of bands
     * \param[in] access Access mode
     */
    template<typename T>
    Dataset(T * data, int width, int length, int bands, GDALAccess access = GA_Update);

    /**
     * Create a read-only dataset describing an existing in-memory array.
     *
     * \param[in] data Pointer to first pixel of first band
     * \param[in] width Number of columns
     * \param[in] length Number of rows
     * \param[in] bands Number of bands
     * \param[in] colstride Stride in bytes between the start of adjacent columns
     * \param[in] rowstride Stride in bytes between the start of adjacent rows
     * \param[in] bandstride Stride in bytes between the start of adjacent bands
     */
    template<typename T>
    Dataset(const T * data,
            int width,
            int length,
            int bands,
            std::size_t colstride,
            std::size_t rowstride,
            std::size_t bandstride);

    /**
     * Create a dataset describing an existing in-memory array.
     *
     * \param[in] data Pointer to first pixel of first band
     * \param[in] width Number of columns
     * \param[in] length Number of rows
     * \param[in] bands Number of bands
     * \param[in] colstride Stride in bytes between the start of adjacent columns
     * \param[in] rowstride Stride in bytes between the start of adjacent rows
     * \param[in] bandstride Stride in bytes between the start of adjacent bands
     * \param[in] access Access mode
     */
    template<typename T>
    Dataset(T * data,
            int width,
            int length,
            int bands,
            std::size_t colstride,
            std::size_t rowstride,
            std::size_t bandstride,
            GDALAccess access = GA_Update);

    /** Access mode */
    GDALAccess access() const { return _access; }

    /** Number of columns */
    int width() const { return _dataset->GetRasterXSize(); }

    /** Number of rows */
    int length() const { return _dataset->GetRasterYSize(); }

    /** Number of bands */
    int bands() const { return _dataset->GetRasterCount(); }

    /** Driver name */
    std::string driver() const { return _dataset->GetDriverName(); }

    /**
     * Fetch raster band.
     *
     * \param[in] band Raster band index (1-based)
     */
    Raster getRaster(int band) const;

    /**
     * Get transform from raster coordinates (pixel, line) to projected
     * coordinates (x, y)
     *
     * If no transform is found, the default (identity) transform is returned.
     */
    GeoTransform getGeoTransform() const;

    /**
     * Set geotransform
     *
     * \throws isce3::except::GDALError if the format does not support this operation
     */
    void setGeoTransform(const GeoTransform &);

    /**
     * Get spatial reference system
     *
     * \throws isce3::except::GDALError if the spatial reference system is unavailable
     */
    isce3::core::ProjectionBase * getProjection() const;

    /**
     * Set spatial reference system
     *
     * \throws isce3::except::GDALError if the format does not support this operation
     */
    void setProjection(const isce3::core::ProjectionBase *);

    /** Left edge of left-most pixel in projection coordinates */
    double x0() const { return getGeoTransform().x0; }

    /** Upper edge of upper-most line in projection coordinates */
    double y0() const { return getGeoTransform().y0; }

    /** Pixel width in projection coordinates */
    double dx() const { return getGeoTransform().dx; }

    /** Line height in projection coordinates */
    double dy() const { return getGeoTransform().dy; }

    /** Get the underlying GDALDataset pointer */
    GDALDataset * get() { return _dataset.get(); }

    /** Get the underlying GDALDataset pointer */
    const GDALDataset * get() const { return _dataset.get(); }

    friend class Raster;

private:
    std::shared_ptr<GDALDataset> _dataset;

    // XXX need to keep track of access mode ourselves rather than use `_dataset->GetAccess()`
    // XXX GDAL silently ignores the requested access mode when using "MEM" driver
    // XXX and sets access mode to GA_Update (https://github.com/OSGeo/gdal/issues/1971)
    GDALAccess _access;
};

}}}

#define ISCE_IO_GDAL_DATASET_ICC
#include "Dataset.icc"
#undef ISCE_IO_GDAL_DATASET_ICC
