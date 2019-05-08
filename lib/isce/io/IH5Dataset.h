//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019


#ifndef __ISCE_IO_IH5DATASET_H__
#define __ISCE_IO_IH5DATASET_H__

#include "gdal_pam.h"
#include "gdal_priv.h"
#include "gdal_rat.h"
#include "IH5.h"


namespace isce {
namespace io {

/************************************************************************/
/*                            IH5Dataset                                */
/************************************************************************/
class IH5RasterBand;

/** IH5 interface to GDAL Dataset to allow 
 *  read/write to HDF5 datasets from ISCE*/
class IH5Dataset final: public GDALDataset
{
    CPL_DISALLOW_COPY_ASSIGN(IH5Dataset)

    friend class IH5RasterBand;

    //Standard stuff expected from all formats
    int bGeoTransformSet; 
    double adfGeoTransform[6];

    char *pszProjection;
    CPLString pszGCPProjection;
    GDAL_GCP *pasGCPList;
    int  nGCPCount; 
    OGRSpatialReference oSRS;

    //IDataSet from Francois's IH5
    isce::io::IDataSet* _dataset;
    H5::DataType nativeType;
    H5::DataType actualType;
    int ndims;
    int dimensions[3];
    int chunks[3];

    protected:
        CPLErr populateFromDataset();

    public:
        /**Empty constructor */
        IH5Dataset(const hid_t &inputds, GDALAccess eAccess);

        /** Destructor */
        virtual ~IH5Dataset();

        virtual const char *GetProjectionRef() override;
        virtual CPLErr SetProjection( const char * ) override;
        virtual int GetGCPCount() override;
        virtual const char *GetGCPProjection() override;
        virtual const GDAL_GCP *GetGCPs() override;
        virtual CPLErr SetGCPs( int nGCPCount, const GDAL_GCP *pasGCPList,
                            const char *pszGCPProjection ) override;

        void *GetInternalHandle (const char *) override;

        virtual CPLErr GetGeoTransform( double *padfTransform ) override;
        virtual CPLErr SetGeoTransform( double * ) override;
        static GDALDataset *Open(GDALOpenInfo *info);
        static int Identify(GDALOpenInfo *info);
};


/************************************************************************/
/*                            IH5RasterBand                             */
/************************************************************************/

/** Raster band of an IH5 Dataset derived from GDALPamRasterBand */
class IH5RasterBand : public GDALPamRasterBand
{
    protected:
        friend class IH5Dataset;
        bool bNoDataSet;
        double dfNoData;

    public:
        IH5RasterBand(IH5Dataset *ds, int band, 
                      GDALDataType eTypeIn);

        virtual ~IH5RasterBand();

        virtual CPLErr IReadBlock( int, int, void * ) override;
        virtual CPLErr IWriteBlock( int, int, void * ) override;
        virtual double GetNoDataValue( int *pbSuccess = nullptr ) override;
        virtual CPLErr SetNoDataValue( double ) override;
};


/** Function to register driver with GDAL */
void GDALRegister_IH5();

}
}

#endif /*  __ISCE_IO_IH5DATASET_H__ */
