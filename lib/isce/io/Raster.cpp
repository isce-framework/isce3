//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Marco Lavalle
// Original code: Joshua Cohen
// Copyright 2018
//

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "Raster.h"


/**
 * @param[in] fname Existing filename
 * @param[in] access GDAL access mode
 *
 * Files are opened with GDALOpenShared*/
isce::io::Raster::Raster(const std::string &fname,   // filename
                         GDALAccess access) {        // GA_ReadOnly or GA_Update

    GDALAllRegister();  // GDAL checks internally if drivers are already loaded
    auto tmp = static_cast<GDALDataset*>(GDALOpenShared(fname.c_str(), access));
    if (tmp == nullptr)
        throw isce::except::RuntimeError(ISCE_SRCINFO(),
                "failed to create GDAL dataset from file '" + fname + "'");
    dataset(tmp);
}


/**
 * @param[in] inputDataset Pointer to an existing dataset*/
isce::io::Raster::Raster(GDALDataset * inputDataset) {
    GDALAllRegister();
    dataset(inputDataset);
}


/**
 * @param[in] fname Filename to create
 * @param[in] width Width of raster image
 * @param[in] length Length of raster image
 * @param[in] numBands Number of bands in raster image
 * @param[in] dtype GDALDataType associated with dataset
 * @param[in] driverName GDAL Driver to use
 *
 * In general, GDAL is used to create dataset. When VRT driver is used, the
 * dataset is interpreted in a special manner - it is assumed that the user
 * expects a flat binary file with a VRT pointing to the data using VRTRawRasterBand*/
isce::io::Raster::Raster(const std::string &fname,          // filename
                         size_t width,                      // number of columns
                         size_t length,                     // number of lines
                         size_t numBands,                   // number of bands
                         GDALDataType dtype,                // band datatype
                         const std::string & driverName) {  // GDAL raster format

    GDALAllRegister();
    GDALDriver * outputDriver = GetGDALDriverManager()->GetDriverByName(driverName.c_str());

    if (driverName == "VRT") {   // if VRT, create empty dataset and add a band, then update.
                                 // Number of bands is forced to 1 for now (numBands is ignored).
                                 // Multi-band VRT can be created by adding band after creation
        dataset( outputDriver->Create (fname.c_str(), width, length, 0, dtype, NULL) );
        addRawBandToVRT( fname, dtype );
        GDALClose( dataset() );
        dataset( static_cast<GDALDataset*>(GDALOpenShared( fname.c_str(), GA_Update )) ); ;
    } else                       // if non-VRT, create dataset using user-defined driver
        dataset( outputDriver->Create (fname.c_str(), width, length, numBands, dtype, NULL) );

}

/**
 * @param[in] fname File name to create
 * @param[in] width Width of raster image
 * @param[in] length Length of raster image
 * @param[in] dtype GDALDataType associated with dataset*/
isce::io::Raster::Raster(const std::string &fname, size_t width, size_t length, GDALDataType dtype) :
    isce::io::Raster(fname, width, length, 1, dtype, isce::io::defaultGDALDriver) {}


/**
 * @param[in] fname File name to create
 * @param[in] rast Reference raster object*/
isce::io::Raster::Raster(const std::string &fname, const Raster &rast) :
    isce::io::Raster(fname, rast.width(), rast.length(), rast.numBands(), rast.dtype()) {}



/**
 * @param[in] rast Source raster.
 *
 * It increments GDAL's reference counter after weak-copying the pointer */
isce::io::Raster::Raster(const Raster &rast) {
    dataset( rast._dataset );
    dataset()->Reference();
}



/**
 * @param[in] fname Output VRT filename to create
 * @param[in] rastVec std::vector of Raster objects*/
isce::io::Raster::Raster(const std::string& fname, const std::vector<Raster>& rastVec) {
    GDALAllRegister();
    GDALDriver * outputDriver = GetGDALDriverManager()->GetDriverByName("VRT");
    dataset(outputDriver->Create (fname.c_str(),
            rastVec.front().width(),
            rastVec.front().length(),
            0,    // bands are added below
            rastVec.front().dtype(),
            NULL) );

    for (auto r : rastVec)     // for each input Raster object in rastVec
        addRasterToVRT( r );
}

/** Uses GDAL's inbuilt OSRFindMatches to determine the EPSG code
 * from the WKT representation of the projection system. This is
 * designed to work with GDAL 2.3+*/
int isce::io::Raster::getEPSG()
{
    //Extract WKT string corresponding to the dataset
    const char* pszProjection = GDALGetProjectionRef(_dataset);

    //If WKT string is not empty
    if (pszProjection == nullptr || strlen(pszProjection) <= 0)
        return -9999;

    //Create a spatial reference object
    OGRSpatialReference hSRS(nullptr);

    //Try to import WKT discovered from dataset
    if (hSRS.importFromWkt(&pszProjection) != 0) {
        std::cout << "Could not interpret following string as a valid wkt projection \n";
        std::cout << pszProjection << "\n";
        return -9998;
    }

    //GDAL 2.3 provides OSRFindMatches which is more robust and thorough.
    //Auto-discovery is only bound to get better.
    int nEntries = 0;
    int* panConfidence = nullptr;

    //This is GDAL 2.3+ way of auto-discovering projection system
    OGRSpatialReferenceH* pahSRS =
        OSRFindMatches(reinterpret_cast<OGRSpatialReferenceH>(
                           const_cast<OGRSpatialReference*>(&hSRS)),
                           nullptr, &nEntries, &panConfidence);

    //If number of matching entries is not positive
    if (nEntries <= 0) {
        std::cout << "Could not find a match in EPSG data base for wkt: \n";
        std::cout << pszProjection << "\n";
        return -9997;
    }

    //If the best match does not have 100% confidence level
    if (panConfidence[0] != 100) {
        std::cout << "Could not find a 100% match in EPSG data base for wkt: \n";
        std::cout << pszProjection << "\n";
        return -9996;
    }

    OGRSpatialReference oSRS = *reinterpret_cast<OGRSpatialReference*>(pahSRS[0]);
    const char* pszAuthorityCode = oSRS.GetAuthorityCode(nullptr);
    int epsgcode = -9999;
    if (pszAuthorityCode)
        epsgcode = std::atoi(pszAuthorityCode);

    OSRFreeSRSArray(pahSRS);
    CPLFree(panConfidence);

    return epsgcode;
}

/**
 * @param[in] epsgcode EPSG code corresponding to projection system
 *
 * GDAL relies on GDAL_DATA environment variable to interpret these codes.
 * Make sure that these are set. */
int isce::io::Raster::setEPSG(int epsgcode)
{
    int status = 1;

    //Create a spatial reference object
    OGRSpatialReference hSRS(nullptr);

    //Try importing from EPSGcode
    if (hSRS.importFromEPSG(epsgcode) == 0)
    {
        char *pszOutput = nullptr;

        //Export to WKT
        hSRS.exportToWkt( &pszOutput);

        //Set projection for dataset
        _dataset->SetProjection(pszOutput);

        CPLFree(pszOutput);

        status = 0;
    }
    else
    {
        //Should use journal logging in future
        std::cout << "Could not interpret EPSG code: " << epsgcode << "\n";
    }

    return status;
}
// Destructor. When GDALOpenShared() is used the dataset is dereferenced
// and closed only if the referenced count is less than 1.
isce::io::Raster::~Raster() {
    GDALClose( _dataset );
}


// end of file
