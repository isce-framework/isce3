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
  dataset( static_cast<GDALDataset*>(GDALOpenShared(fname.c_str(), access)) );
}



/**
 * @param[in] fname Existing filename to be opened in ReadOnly mode*/
isce::io::Raster::Raster(const std::string &fname) :
  isce::io::Raster(fname, GA_ReadOnly) {}


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
  }

  else                         // if non-VRT, create dataset using user-defined driver    
    dataset( outputDriver->Create (fname.c_str(), width, length, numBands, dtype, NULL) );
  
}



/** 
 * @param[in] fname File name to create
 * @param[in] width Width of raster image
 * @param[in] length Length of raster image
 * @param[in] numBands Number of bands in raster image
 * @param[in] dtype GDAL Datatype associated with dataset*/
isce::io::Raster::Raster(const std::string &fname, size_t width, size_t length, size_t numBands, GDALDataType dtype) :
  isce::io::Raster(fname, width, length, numBands, dtype, isce::io::defaultGDALDriver) {}



/**
 * @param[in] fname File name to create
 * @param[in] width Width of raster image
 * @param[in] length Length of raster image
 * @param[in] numBands Number of bands in raster image*/
isce::io::Raster::Raster(const std::string &fname, size_t width, size_t length, size_t numBands) :
  isce::io::Raster(fname, width, length, numBands, isce::io::defaultGDALDataType) {}



/**
 * @param[in] fname File name to create
 * @param[in] width Width of raster image
 * @param[in] length Length of raster image
 * @param[in] dtype GDALDataType associated with dataset*/
isce::io::Raster::Raster(const std::string &fname, size_t width, size_t length, GDALDataType dtype) :
  isce::io::Raster(fname, width, length, 1, dtype, isce::io::defaultGDALDriver) {}


/** 
 * @param[in] fname File name to create
 * @param[in] width Width of raster image
 * @param[in] length Length of raster image*/
isce::io::Raster::Raster(const std::string &fname, size_t width, size_t length) :
  isce::io::Raster(fname, width, length, 1) {}



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
  dataset( outputDriver->Create (fname.c_str(),
				 rastVec.front().width(),
				 rastVec.front().length(),
				 0,    // bands are added below
				 rastVec.front().dtype(),
				 NULL) );  

  for ( auto r : rastVec)     // for each input Raster object in rastVec
    addRasterToVRT( r );
}

/** Uses GDAL's inbuilt OSRFindMatches to determine the EPSG code
 * from the WKT representation of the projection system. This is 
 * designed to work with GDAL 2.3+*/
int isce::io::Raster::getEPSG()
{
    int epsgcode = -9999;

    //Extract WKT string corresponding to the dataset
    const char* pszProjection = GDALGetProjectionRef(_dataset);
    
    
    //If WKT string is not empty
    if ((pszProjection != nullptr) && strlen(pszProjection)>0)
    {
        //Create a spatial reference object
        OGRSpatialReference hSRS(nullptr);

	//These two lines can be deleted if we enforce GDAL >= 2.3
        //This dance to move from const char* to char* for older versions
        int lenstr = strlen(pszProjection);             //Does not include null char
        char* pszProjectionTmp = new char[lenstr+1];    //Assign space including null char
        strncpy(pszProjectionTmp, pszProjection, lenstr); //Copy valid data
        pszProjectionTmp[lenstr] = '\0';                //Ensure null char at end

        //Assign to temp variable as importFromWkt increments pointer
        //We keep track of assigned memorty in pszProjectionTmp for delete later
        char *ptr = pszProjectionTmp;
	
        //Try to import WKT discovered from dataset
	//if ( hSRS.importFromWkt(&pszProjection) == 0 )  // use char* if we enforce GDAL >= 2.3
        if ( hSRS.importFromWkt( &ptr ) == 0 )
        {

            //This part of the code is for features below GDAL 2.3
            //We are forced to use AutoIdentifyEPSG which is not complete
            
#if GDAL_VERSION_MINOR < 3
            if (hSRS.AutoIdentifyEPSG() == 0) 
            {
                std::string authorityName, authorityCode;

                //For geographic system, authority is provided by GEOGCS
                if (hSRS.IsGeographic())
                {
                    std::cout << "Geographic \n";
                    authorityName.assign("GEOGCS");
                }
                //For projected system, authority is provided by PROJCS
                else if (hSRS.IsProjected())
                {
                    std::cout << "Projected \n";
                    authorityName.assign("PROJCS");
                }
                else if (hSRS.IsLocal())
                {
                    throw "EPSG codes associated with local systems are not supported";
                }
                else
                {
                    throw "Unsupported coordinate system encountered";
                }

                //Use authority name to extract authority code
                const char* ptr = authorityName.c_str();
                authorityCode.assign( hSRS.GetAuthorityCode(ptr));
                epsgcode = std::atoi(authorityCode.c_str());
            }
            else
            {
                epsgcode = -9997;
                //Should be captured by journal in future
                std::cout << "Could not auto-identify EPSG for wkt representation: \n";
                std::cout << pszProjection << "\n";
            }
#else
            //GDAL 2.3 provides OSRFindMatches which is more robust and thorough.
            //Auto-discovery is only bound to get better. 
            //GDAL 2.3 (May 2018) is still relatively new. 
            //In about a year's time we will be able
            //to deprecate the above part.
            int nEntries = 0;
            int* panConfidence = nullptr;
            OGRSpatialReferenceH* pahSRS = nullptr;

            //This is GDAL 2.3+ way of auto-discovering projection system
            pahSRS = OSRFindMatches( reinterpret_cast<OGRSpatialReferenceH>(
                                        const_cast<OGRSpatialReference*>(&hSRS)),
                                    nullptr, &nEntries, &panConfidence);

            //If number of matching entries is greater than 0
            if (nEntries > 0)
            {
                //If the best match has 100% confidence level
                if (panConfidence[0] == 100)
                {
                    OGRSpatialReference oSRS = *reinterpret_cast<OGRSpatialReference*>(pahSRS[0]);
                    const char* pszAuthorityCode = oSRS.GetAuthorityCode(nullptr);
                    if (pszAuthorityCode)
                        epsgcode = std::atoi(pszAuthorityCode);
                }
                else
                {
                    epsgcode = -9996;
                    std::cout << "Could not find a 100% match in EPSG data base for wkt: \n";
                    std::cout << pszProjection << "\n";
                }
            }
            else
            {
                epsgcode = -9997;
                std::cout << "Could not find a match in EPSG data base for wkt: \n";
                std::cout << pszProjection << "\n";
            }

            OSRFreeSRSArray(pahSRS);
            CPLFree(panConfidence);
#endif
        }
        else
        {
            epsgcode = -9998;
            std::cout << "Could not interpret following string as a valid wkt projection \n";
            std::cout << pszProjection << "\n";
        }
        delete [] pszProjectionTmp;
    }

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
