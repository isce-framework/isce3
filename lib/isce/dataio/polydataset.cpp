//
// Author: Joshua Cohen
// Copyright 2018
//
// Author's Note: The following class is designed with the general GDAL and GDALDataset style
//                in mind, with the intention to implement a GDAL Driver for the Poly2D class. This
//                is so that we can abstract the RasterIO functions for the Image interface to deal
//                with image files, polynomial generators, and lookup tables as valid data sources.
//

#ifndef __ISCE_DATAIO_POLYDATASET_CPP__
#define __ISCE_DATAIO_POLYDATASET_CPP__

#include "Poly2d.h"
#include <gdal_priv.h>

class PolyRasterBand;

class PolyDataset : public GDALDataset {
    friend class PolyRasterBand;

    isce::core::Poly2d poly;

    public:
        PolyDataset(isce::core::Poly2d &p) : poly(p) {}
        ~PolyDataset();

        static GDALDataset* Open(isce::core::Poly2d&,size_t,size_t);
        static int Identify(isce::core::Poly2d&);
        //CPLErr GetGeoTransform(double *padfTransform);
        inline void setLength(size_t l) { nRasterYSize = l; }
        inline void setWidth(size_t w) { nRasterXSize = w; }
        inline void setDimensions(size_t,size_t);
};

GDALDataset* PolyDataset::Open(isce::core::Poly2d &p, size_t length=128, size_t width=128) {
    if (!Identify(p)) return nullptr;

    PolyDataset *poDS = new PolyDataset(p);
    poDS->nRasterXSize = width;
    poDS->nRasterYSize = length;
    poDS->SetBand(1, new PolyRasterBand(poDS));
    poDS->SetDescription("Poly2d-generated Image");
    return poDS;
}

int PolyDataset::Identify(isce::core::Poly2d &p) {
    // Don't allow empty polynomial images
    if (p.coeffs.size() == 0) return FALSE;
    // Don't allow mis-sized polynomials (shouldn't be possible given Poly2d's design, but just in
    // case...)
    if (((p.rangeOrder+1) * (p.azimuthOrder+1)) != p.coeffs.size()) return FALSE;
    return TRUE;
}

inline void PolyDataset::setDimensions(size_t length, size_t width) {
    setLength(length);
    setWidth(width);
}

class PolyRasterBand : public GDALRasterBand {
    friend class PolyDataset;

    public:
        PolyRasterBand(PolyDataset*);
        ~PolyRasterBand();

        virtual CPLErr IReadBlock(int,int,void*);
};

PolyRasterBand::PolyRasterBand(PolyDataset *poDSIn) {
    poDS = poDSIn;
    // Poly Images will only have 1 band (avoid confusion)
    nBand = 1;
    // Poly Images will (due to their coefficients being double-precision) always be doubles
    eDataType = GDT_Float64;
    nBlockXSize = poDSIn->GetRasterXSize();
    // At GDAL's recommendation, non-tiled datasets have a natural block size of one scanline
    nBlockYSize = 1;
}

CPLErr PolyRasterBand::IReadBlock(int nBlockXOff, int nBlockYOff, void *pImage) {
    // Standard GDAL practice
    PolyDataset *poGDS = static_cast<PolyDataset*>(poDS);
    // Width offset (number of blocks x width of block)
    int xOff = nBlockXOff * nBlockXSize;
    // Fill pImage buffer with poly-calculated values
    for (int idx=0; idx<nBlockXSize; ++idx) {
        static_cast<double*>(pImage)[idx] = poGDS->poly.eval(nBlockYOff, xOff + idx);
    }
    return CE_None;
}

/*
    Still need to write RasterIO overload here...
*/

void GDALRegister_POLY() {
    if (!GDAL_CHECK_VERSION("POLY")) return;
    if (GDALGetDriverByName("POLY") != nullptr) return;
    
    GDALDriver *poDriver = new GDALDriver();
    poDriver->SetDescription("POLY");
    poDriver->SetMetadataItem(GDAL_DCAP_RASTER, "YES");
    poDriver->SetMetadataItem(GDAL_DMD_LONGNAME, "ISCE Poly2d-generated Image (virtual).");
    poDriver->pfnOpen = PolyDataset::Open;
    poDriver->pfnIdentify = PolyDataset::Identify;

    GetGDALDriverManager()->RegisterDriver(poDriver);
}

#endif
