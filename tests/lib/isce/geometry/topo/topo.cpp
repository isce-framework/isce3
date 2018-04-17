//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2018
//

#include <iostream>
#include <complex>
#include <string>
#include <sstream>
#include <fstream>
#include <gtest/gtest.h>

// isce::core
#include "isce/core/Constants.h"
#include "isce/core/Raster.h"
#include "isce/core/Serialization.h"

// isce::geometry
#include "isce/geometry/Serialization.h"
#include "isce/geometry/Topo.h"

// Declaration for utility function to read metadata stream from VRT
std::stringstream streamFromVRT(const char * filename, int bandNum=1);

// Declare the pixel function for getting DEM height
CPLErr evaluateHeight(void **papoSources, int nSources, void *pData, int nXSize, int nYSize,
    GDALDataType eSrcType, GDALDataType eBufType, int nPixelSpace, int nLineSpace);

TEST(TopoTest, RunTopo) {

    // Instantiate isce::core objects
    isce::core::Poly2d doppler;
    isce::core::Orbit orbit;
    isce::core::Ellipsoid ellipsoid;
    isce::core::Metadata meta;

    // Load metadata
    std::stringstream metastream = streamFromVRT("../../data/envisat.slc.vrt");
    {
    cereal::XMLInputArchive archive(metastream);
    archive(cereal::make_nvp("Orbit", orbit),
            cereal::make_nvp("SkewDoppler", doppler),
            cereal::make_nvp("Ellipsoid", ellipsoid),
            cereal::make_nvp("Radar", meta));
    }

    // Create topo instance
    isce::geometry::Topo topo(ellipsoid, orbit, meta);

    // Load topo processing parameters to finish configuration
    std::ifstream xmlfid("../../data/topo.xml", std::ios::in);
    {
    cereal::XMLInputArchive archive(xmlfid);
    archive(cereal::make_nvp("Topo", topo));
    }

    // Let GDAL know about pixel function for virtual DEM
    GDALAddDerivedBandPixelFunc("evaluateHeight", evaluateHeight);
    // Register drivers
    GDALAllRegister();

    // Open DEM raster
    isce::core::Raster demRaster("../../data/constantDEM.vrt");

    // Run topo
    topo.topo(demRaster, doppler, "output");

}

// Read metadata from a VRT file and return a stringstream object
std::stringstream streamFromVRT(const char * filename, int bandNum) {
    // Register GDAL drivers
    GDALAllRegister();
    // Open the VRT dataset
    GDALDataset * dataset = (GDALDataset *) GDALOpen(filename, GA_ReadOnly);
    if (dataset == NULL) {
        std::cout << "Cannot open dataset " << filename << std::endl;
        exit(1);
    }
    // Read the metadata
    char **metadata_str = dataset->GetRasterBand(bandNum)->GetMetadata("xml:isce");
    // The cereal-relevant XML is the first element in the list
    std::string meta{metadata_str[0]};
    // Close the VRT dataset
    GDALClose(dataset);
    // Convert to stream
    std::stringstream metastream;
    metastream << meta;
    // All done
    return metastream;
}

// Simply return a value of 0 for a virtual raster
CPLErr evaluateHeight(void **papoSources, int nSources, void *pData, int nXSize, int nYSize,
    GDALDataType eSrcType, GDALDataType eBufType, int nPixelSpace, int nLineSpace) {
    // Fill with zeros
    for (int i = 0; i < nYSize; ++i) {
        for (int j = 0; j < nXSize; ++j) {
            float pix_value = 0.0;
            GDALCopyWords(&pix_value, GDT_Float32, 0, ((GByte *) pData) + nLineSpace * i
                + j * nPixelSpace, eBufType, nPixelSpace, 1);
        }
    }
    // Done
    return CE_None;
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
