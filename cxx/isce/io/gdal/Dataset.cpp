#include "Dataset.h"

#include <array>

#include <isce/except/Error.h>
#include <isce/io/IH5Dataset.h>

#include "Raster.h"
#include "SpatialReference.h"

namespace isce { namespace io { namespace gdal {

static
void registerDrivers()
{
    static bool registered(false);
    if (!registered) {
        // register GDAL drivers (only needs to be done once)
        GDALAllRegister();
        GDALRegister_IH5();
        registered = true;
    }
}

inline
std::string toString(GDALAccess access)
{
    switch (access) {
        case GA_ReadOnly : return "GA_ReadOnly";
        case GA_Update   : return "GA_Update";
    }

    throw isce::except::RuntimeError(ISCE_SRCINFO(), "unexpected GDALAccess value");
}

static
GDALDataset * openDataset(const std::string & path, GDALAccess access)
{
    registerDrivers();

    // open file
    GDALDatasetH handle = GDALOpen(path.c_str(), access);
    if (!handle) {
        std::string errmsg = "unable to open GDAL dataset '" + path + "' "
            "with access '" + toString(access) + "'";
        throw isce::except::GDALError(ISCE_SRCINFO(), errmsg);
    }

    // cast to GDALDataset *
    return GDALDataset::FromHandle(handle);
}

static
GDALDataset * createDataset(
        const std::string & path,
        int width,
        int length,
        int bands,
        GDALDataType datatype,
        const std::string & driver_name)
{
    registerDrivers();

    // get driver
    GDALDriver * driver = GetGDALDriverManager()->GetDriverByName(driver_name.c_str());
    if (!driver) {
        std::string errmsg = "no match found for GDAL driver '" + driver_name + "'";
        throw isce::except::GDALError(ISCE_SRCINFO(), errmsg);
    }

    // create dataset
    GDALDataset * dataset = driver->Create(path.c_str(), width, length, bands, datatype, nullptr);
    if (!dataset) {
        std::string errmsg = "unable to create GDAL dataset at '" + path + "' using driver '" + driver_name + "'";
        throw isce::except::GDALError(ISCE_SRCINFO(), errmsg);
    }

    return dataset;
}

static
GDALDataset * copyDataset(const std::string & path,
                          GDALDataset * src,
                          const std::string & driver_name)
{
    registerDrivers();

    // get driver
    GDALDriver * driver = GetGDALDriverManager()->GetDriverByName(driver_name.c_str());
    if (!driver) {
        std::string errmsg = "no match found for GDAL driver '" + driver_name + "'";
        throw isce::except::GDALError(ISCE_SRCINFO(), errmsg);
    }

    // create copy
    GDALDataset * dataset = driver->CreateCopy(path.c_str(), src, 0, nullptr, nullptr, nullptr);
    if (!dataset) {
        std::string errmsg = "unable to create GDAL dataset at '" + path + "' using driver '" + driver_name + "'";
        throw isce::except::GDALError(ISCE_SRCINFO(), errmsg);
    }

    return dataset;
}

Dataset::Dataset(const std::string & path, GDALAccess access)
:
    _dataset(openDataset(path, access)),
    _access(access)
{}

Dataset::Dataset(const IDataSet & dataset, GDALAccess access)
:
    Dataset(dataset.toGDAL(), access)
{}

Dataset::Dataset(const std::string & path,
                 int width,
                 int length,
                 int bands,
                 GDALDataType datatype,
                 const std::string & driver)
:
    _dataset(createDataset(path, width, length, bands, datatype, driver)),
    _access(GA_Update)
{}

Dataset::Dataset(const std::string & path, const Dataset & src)
:
    Dataset(path, src, src.driver())
{}

Dataset::Dataset(const std::string & path, const Dataset & src, const std::string & driver)
:
    _dataset(copyDataset(path, const_cast<GDALDataset *>(src.get()), driver)),
    _access(_dataset->GetAccess())
{}

Raster Dataset::getRaster(int band) const
{
    // check that raster band is valid
    if (band < 1 || band > bands()) {
        std::string errmsg = "raster band index (" + std::to_string(band) + ") is out of range";
        throw isce::except::OutOfRange(ISCE_SRCINFO(), errmsg);
    }

    return Raster(*this, band);
}

GeoTransform Dataset::getGeoTransform() const
{
    std::array<double, 6> coeffs;
    CPLErr status = _dataset->GetGeoTransform(coeffs.data());
    return (status == CE_None) ? GeoTransform(coeffs) : GeoTransform();
}

void Dataset::setGeoTransform(const GeoTransform & transform)
{
    if (access() == GA_ReadOnly) {
        throw isce::except::RuntimeError(ISCE_SRCINFO(), "attempted to modify read-only dataset");
    }

    std::array<double, 6> coeffs = transform.getCoeffs();
    CPLErr status = _dataset->SetGeoTransform(coeffs.data());
    if (status != CE_None) {
        throw isce::except::GDALError(ISCE_SRCINFO(), "unable to set geotransform");
    }
}

isce::core::ProjectionBase * Dataset::getProjection() const
{
    // get string defining spatial reference system
    const char * tmp = _dataset->GetProjectionRef();
    if (!tmp) {
        throw isce::except::GDALError(ISCE_SRCINFO(), "unable to fetch projection reference");
    }
    std::string wkt(tmp);

    // get EPSG code
    SpatialReference srs(wkt);
    int epsg = srs.toEPSG();

    return isce::core::createProj(epsg);
}

void Dataset::setProjection(const isce::core::ProjectionBase * proj)
{
    if (!proj) {
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), "projection pointer may not be null");
    }

    if (access() == GA_ReadOnly) {
        throw isce::except::RuntimeError(ISCE_SRCINFO(), "attempted to modify read-only dataset");
    }

    // get spatial reference via EPSG code
    int epsg = proj->code();
    SpatialReference srs(epsg);

    // get WKT representation
    std::string wkt = srs.toWKT();

    // set raster spatial reference system
    _dataset->SetProjection(wkt.c_str());
}

}}}
