"""
collection of functions to prepare HDF5
"""

import os

import h5py
import numpy as np
from osgeo import osr

from pybind_nisar.h5 import cp_h5_meta_data
from pybind_nisar.products.readers import SLC
import pybind_isce3 as isce3


def run(cfg, dst):
    cp_geocode_meta(cfg, dst)
    prep_ds(cfg, dst)


def cp_geocode_meta(cfg, dst):
    '''
    Copies shared data from src HDF5 to GSLC dst

    Parameters:
    -----------
    cfg : dict
        Run configuration
    dst : str
        Name of destination node where data is to be copied
    '''

    # unpack
    output_hdf5 = cfg['ProductPathGroup']['SASOutputFile']
    input_hdf5 = cfg['InputFileGroup']['InputFilePath']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    # rm anything and start from scratch
    try:
        os.remove(output_hdf5)
    except FileNotFoundError:
        pass

    slc = SLC(hdf5file=input_hdf5)

    # prelim setup
    common_parent_path = 'science/LSAR'
    src_h5 = h5py.File(input_hdf5, 'r', libver='latest', swmr=True)
    src_meta_path = slc.MetadataPath
    dst_h5 = h5py.File(output_hdf5, 'w', libver='latest', swmr=True)
    dst_meta_path = f'{common_parent_path}/{dst}/metadata'

    # simple copies of identification, metadata/orbit, metadata/attitude groups
    cp_h5_meta_data(src_h5, dst_h5, os.path.join(common_parent_path, 'identification'))
    ident = dst_h5[os.path.join(common_parent_path, 'identification')]
    dset = ident.create_dataset('isGeocoded', data=np.string_("True"))
    desc = f"Flag to indicate radar geometry or geocoded product"
    dset.attrs["description"] = np.string_(desc)

    # copy orbit information group
    cp_h5_meta_data(src_h5, dst_h5, f'{src_meta_path}/orbit',
        f'{dst_meta_path}/orbit')

    # copy attitudue information group
    cp_h5_meta_data(src_h5, dst_h5, f'{src_meta_path}/attitude',
        f'{dst_meta_path}/attitude')

    # copy calibration information group
    for freq in freq_pols.keys():
        frequency = f'frequency{freq}'
        pol_list = freq_pols[freq]
        if pol_list is None:
            continue
        for polarization in pol_list:
            cp_h5_meta_data(src_h5, dst_h5,
                f'{src_meta_path}/calibrationInformation/{frequency}/{polarization}',
                f'{dst_meta_path}/calibrationInformation/{frequency}/{polarization}')

    # copy processing information group
    cp_h5_meta_data(src_h5, dst_h5,
                    f'{src_meta_path}/processingInformation',
                    f'{dst_meta_path}/processingInformation',
                    excludes=['l0bGranules', 'demFiles', 'zeroDopplerTime', 'slantRange'])

    # copy radar grid information group
    cp_h5_meta_data(src_h5, dst_h5,
                    f'{src_meta_path}/geolocationGrid',
                    f'{dst_meta_path}/radarGrid',
                    renames={'coordinateX':'xCoordinates',
                        'coordinateY':'yCoordinates',
                        'zeroDopplerTime':'zeroDopplerAzimuthTime'})
    
    # copy SLC/COV meta data
    for freq in ['A', 'B']:
        ds_ref = f'{slc.SwathPath}/frequency{freq}'
        if ds_ref not in src_h5:
            continue
        cp_h5_meta_data(src_h5, dst_h5, ds_ref,
                os.path.join(common_parent_path, f'{dst}/grids/frequency{freq}'),
                excludes=['acquiredCenterFrequency', 'acquiredAzimuthBandwidth', 
                    'acquiredRangeBandwidth', 'nominalAcquisitionPRF', 'slantRange',
                    'sceneCenterAlongTrackSpacing', 'sceneCenterGroundRangeSpacing',
                    'HH', 'HV', 'VH', 'VV', 'RH', 'RV',
                    'validSamplesSubSwath1', 'validSamplesSubSwath2',
                    'validSamplesSubSwath3', 'validSamplesSubSwath4',
                    'listOfPolarizations'],
                renames={'processedCenterFrequency':'centerFrequency',
                    'processedAzimuthBandwidth':'azimuthBandwidth',
                    'processedRangeBandwidth':'rangeBandwidth'})

    input_grp = dst_h5[f'{dst_meta_path}/processingInformation/inputs']
    dset = input_grp.create_dataset("l1SlcGranules", data=np.string_([input_hdf5]))
    desc = f"List of input L1 products used"
    dset.attrs["description"] = np.string_(desc)

    dst_h5.close()
    src_h5.close()

def prep_ds(cfg, dst):
    '''
    prepare GSLC dataset rasters and ancillary data
    XXX do this with GCOV?
    '''
    # unpack
    common_parent_path = 'science/LSAR'
    output_hdf5 = cfg['ProductPathGroup']['SASOutputFile']
    input_hdf5 = cfg['InputFileGroup']['InputFilePath'][0]
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    dst_h5 = h5py.File(output_hdf5, 'a', libver='latest', swmr=True)

    ctype = h5py.h5t.py_create(np.complex64)
    ctype.commit(dst_h5['/'].id, np.string_('complex64'))

    # create the datasets in the output hdf5
    geogrids = cfg['processing']['geocode']['geogrids']
    for freq in freq_pols.keys():
        pol_list = freq_pols[freq]
        shape=(geogrids[freq].length, geogrids[freq].width)
        dst_parent_path = os.path.join(common_parent_path, f'{dst}/grids/frequency{freq}')

        _add_geo_info(dst_h5, dst, common_parent_path, freq, geogrids[freq])

        # GSLC specfics datasets
        if dst == 'GSLC':
            for polarization in pol_list:
                dst_grp = dst_h5[dst_parent_path]
                _create_datasets(dst_grp,
                        freq, polarization, shape, chunks=(128, 128))

        # set GCOV polarization values (diagonal values only)
        if dst == 'GCOV':
            pol_list = [(p+p).upper() for p in pol_list]

        _add_polarization_list(dst_h5, dst, common_parent_path, freq, pol_list)

    dst_h5.close()


def _create_datasets(dst_grp, frequency, polarization, shape, chunks=(128, 128)):
    '''
    create 
    '''

    # XXX make type a param
    # XX should be in pybind_nisar.h5?
    ctype = h5py.h5t.py_create(np.complex64)

    if chunks[0]<shape[0] and chunks[1]<shape[1]:
        ds = dst_grp.create_dataset(polarization, dtype=ctype, shape=shape, chunks=chunks)
    else:
        ds = dst_grp.create_dataset(polarization, dtype=ctype, shape=shape)

    ds.attrs['description'] = np.string_(f'Geocoded SLC for {polarization} channel')
    ds.attrs['units'] = np.string_('')

    ds.attrs['grid_mapping'] = np.string_("projection")

    return None


def _add_polarization_list(dst_h5, dst, common_parent_path, frequency, pols):
    '''

    '''
    dataset_path = os.path.join(common_parent_path, f'{dst}/grids/frequency{frequency}')
    grp = dst_h5[dataset_path]
    name = "listOfPolarizations"
    polsArray = np.array(pols, dtype="S2")
    dset = grp.create_dataset(name, data=polsArray)
    desc = f"List of polarization layers with frequency{frequency}"
    dset.attrs["description"] = np.string_(desc)

    return None


def _add_geo_info(hdf5_obj, dst, common_parent_path, frequency, geo_grid):

    epsg_code = geo_grid.epsg

    dx = geo_grid.spacing_x
    x0 = geo_grid.start_x + 0.5*dx
    xf = x0 + geo_grid.width*dx
    x_vect = np.arange(x0, xf, dx, dtype=np.float64)

    dy = geo_grid.spacing_y
    y0 = geo_grid.start_y + 0.5*dy
    yf = y0 + geo_grid.length*dy
    y_vect = np.arange(y0, yf, dy, dtype=np.float64)

    hdf5_obj.attrs['Conventions'] = np.string_("CF-1.8")
    root_ds = os.path.join(common_parent_path, dst, f'grids/frequency{frequency}')

    # xCoordinates
    h5_ds = os.path.join(root_ds, 'xCoordinates') # float64
    xds = hdf5_obj.create_dataset(h5_ds, data=x_vect)

    # yCoordinates
    h5_ds = os.path.join(root_ds, 'yCoordinates') # float64
    yds = hdf5_obj.create_dataset(h5_ds, data=y_vect)

    try:
        for _ds in [xds, yds]:
            _ds.make_scale()
    except AttributeError:
        pass

    #Associate grid mapping with data - projection created later
    h5_ds = os.path.join(root_ds, "projection")

    #Set up osr for wkt
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)

    ###Create a new single int dataset for projections
    projds = hdf5_obj.create_dataset(h5_ds, (), dtype='i')
    projds[()] = epsg_code

    #h5_ds_list.append(h5_ds)

    ##WGS84 ellipsoid
    projds.attrs['semi_major_axis'] = 6378137.0
    projds.attrs['inverse_flattening'] = 298.257223563
    projds.attrs['ellipsoid'] = np.string_("WGS84")

    ##Additional fields
    projds.attrs['epsg_code'] = epsg_code

    ##CF 1.7+ requires this attribute to be named "crs_wkt"
    ##spatial_ref is old GDAL way. Using that for testing only.
    ##For NISAR replace with "crs_wkt"
    projds.attrs['spatial_ref'] = np.string_(srs.ExportToWkt())

    ##Here we have handcoded the attributes for the different cases
    ##Recommended method is to use pyproj.CRS.to_cf() as shown above
    ##To get complete set of attributes.

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg_code)

    projds.attrs['grid_mapping_name'] = sr.GetName()

    ###Set up units
    ###Geodetic latitude / longitude
    if epsg_code == 4326:
        #Set up grid mapping
        projds.attrs['longitude_of_prime_meridian'] = 0.0

        #Setup units for x and y
        xds.attrs['standard_name'] = np.string_("longitude")
        xds.attrs['units'] = np.string_("degrees_east")

        yds.attrs['standard_name'] = np.string_("latitude")
        yds.attrs['units'] = np.string_("degrees_north")
    else:
        ### UTM zones
        if ((epsg_code > 32600 and
                   epsg_code < 32661) or
                  (epsg_code > 32700 and
                   epsg_code < 32761)):
            #Set up grid mapping
            projds.attrs['utm_zone_number'] = epsg_code % 100

        ### Polar Stereo North
        elif epsg_code == 3413:
            #Set up grid mapping
            projds.attrs['standard_parallel'] = 70.0
            projds.attrs['straight_vertical_longitude_from_pole'] = -45.0

        ### Polar Stereo south
        elif epsg_code == 3031:
            #Set up grid mapping
            projds.attrs['standard_parallel'] = -71.0
            projds.attrs['straight_vertical_longitude_from_pole'] = 0.0

        ### EASE 2 for soil moisture L3
        elif epsg_code == 6933:
            #Set up grid mapping
            projds.attrs['longitude_of_central_meridian'] = 0.0
            projds.attrs['standard_parallel'] = 30.0

        ### Europe Equal Area for Deformation map (to be implemented in isce3)
        elif epsg_code == 3035:
            #Set up grid mapping
            projds.attrs['standard_parallel'] = -71.0
            projds.attrs['straight_vertical_longitude_from_pole'] = 0.0

        else:
            raise NotImplementedError(f'EPSG {epsg_code} waiting for implementation / not supported in ISCE3')

        #Setup common parameters
        xds.attrs['standard_name'] = np.string_("projection_x_coordinate")
        xds.attrs['long_name'] = np.string_("x coordinate of projection")
        xds.attrs['units'] = np.string_("m")

        yds.attrs['standard_name'] = np.string_("projection_y_coordinate")
        yds.attrs['long_name'] = np.string_("y coordinate of projection")
        yds.attrs['units'] = np.string_("m")

        projds.attrs['false_easting'] = sr.GetProjParm(osr.SRS_PP_FALSE_EASTING)
        projds.attrs['false_northing'] = sr.GetProjParm(osr.SRS_PP_FALSE_NORTHING)

        projds.attrs['latitude_of_projection_origin'] = sr.GetProjParm(osr.SRS_PP_LATITUDE_OF_ORIGIN)
        projds.attrs['longitude_of_projection_origin'] = sr.GetProjParm(osr.SRS_PP_LONGITUDE_OF_ORIGIN)
