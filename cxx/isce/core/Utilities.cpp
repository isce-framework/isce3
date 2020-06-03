#include "Utilities.h"

#include <gdal_priv.h>

std::string isce::core::stringFromVRT(const char * filename, int bandNum)
{
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

    // All done
    return meta;
}
