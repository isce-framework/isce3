#!/usr/bin/env python3 #

import h5py
import os
from nisar.h5 import cp_h5_meta_data
import numpy as np
import isce3.extensions.isceextension as temp_isce3

def runPrepHDF5(self):
    '''
    Copies shared data from RSLC HDF5 to GSLC HDF5

    Parameters:
    -----------
    path_src : str
        Full path to source HDF5 file
    path_dst : str
        Full path to destination HDF5 file
    '''

    state = self.state

    # prelim setup
    common_parent_path = 'science/LSAR'
    src_h5 = h5py.File(state.input_hdf5, 'r')

    # rm anything and start from scratch
    try:
        os.remove(state.output_hdf5)
    except FileNotFoundError:
        pass

    dst_h5 = h5py.File(state.output_hdf5, 'w')

    # simple copies of identification, metadata/orbit, metadata/attitude groups
    cp_h5_meta_data(src_h5, dst_h5, os.path.join(common_parent_path, 'identification'))
    cp_h5_meta_data(src_h5, dst_h5, 
                    os.path.join(common_parent_path, 'SLC/metadata/orbit'),
                    os.path.join(common_parent_path, 'GSLC/metadata/orbit'))

    cp_h5_meta_data(src_h5, dst_h5, 
                    os.path.join(common_parent_path, 'SLC/metadata/attitude'),
                    os.path.join(common_parent_path, 'GSLC/metadata/attitude'))

    # copy calibration information group
    cp_h5_meta_data(src_h5, dst_h5,
            os.path.join(common_parent_path, 'SLC/metadata/calibrationInformation'),
            os.path.join(common_parent_path, 'GSLC/metadata/calibrationInformation'),
            excludes=['zeroDopplerTime', 'slantRange'])
                
    # copy processing information group
    cp_h5_meta_data(src_h5, dst_h5,
            os.path.join(common_parent_path, 'SLC/metadata/processingInformation'),
            os.path.join(common_parent_path, 'GSLC/metadata/processingInformation'),
            excludes=['l0bGranules', 'demFiles', 'zeroDopplerTime', 'slantRange'])

    # copy radar grid information group
    cp_h5_meta_data(src_h5, dst_h5,
            os.path.join(common_parent_path, 'SLC/metadata/geolocationGrid'),
            os.path.join(common_parent_path, 'GSLC/metadata/radarGrid'),
            renames={'coordinateX':'xCoordinates',
                'coordinateY':'yCoordinates',
                'zeroDopplerTime':'zeroDopplerAzimuthTime'})

    # copy radar imagery group; assumming shared data
    # XXX option0: to be replaced with actual gcov code
    # XXX option1: do not write GSLC data here; GSLC rasters can be appended to the GSLC HDF5
    for freq in ['A', 'B']:
        cp_h5_meta_data(src_h5, dst_h5,
                os.path.join(common_parent_path, f'SLC/swaths/frequency{freq}'),
                os.path.join(common_parent_path, f'GSLC/grids/frequency{freq}'),
                excludes=['acquiredCenterFreqeuncy', 'acquiredAzimuthBandwidth', 
                    'acquiredRangeBandwidth', 'nominalAcquisitionPRF', 'slantRange',
                    'sceneCenterAlongTrackSpacing', 'sceneCenterGroundRangeSpacing',
                    'HH', 'HV', 'VH', 'VV', 'RH', 'RV',
                    'validSamplesSubSwath1', 'validSamplesSubSwath2',
                    'validSamplesSubSwath3', 'validSamplesSubSwath4'],
                renames={'processedCenterFrequency':'centerFrequency',
                    'processedAzimuthBandwidth':'azimuthBandwidth',
                    'processedRangeBandwidth':'rangeBandwidth'})


    self.geogrid_dict = {}
    for freq in state.subset_dict.keys():
        frequency = "frequency{}".format(freq)
        self.geogrid_dict[frequency] = _createGeoGrid(self.userconfig, frequency)
        pol_list = state.subset_dict[freq]
        shape=(self.geogrid_dict[frequency].length, self.geogrid_dict[frequency].width)

        for polarization in pol_list:
            _createDatasets(dst_h5, common_parent_path, frequency, polarization, shape, chunks=(128, 128))
    
    dst_h5.close()

def _createGeoGrid(userconfig, frequency):
    
    # For production we only fix epsgcode and snap value and will 
    # rely on the rslc product metadta to compute the bounding box of the geocoded products
    # there is a place holder in SLC product for compute Bounding box
    # when that method is populated we should be able to simply say
    # bbox = self.slc_obj.computeBoundingBox(epsg=state.epsg)

    #for now let's rely on the run config input  
    x_start = userconfig['processing']['geocode']['top_left']['x_abs']
    y_start = userconfig['processing']['geocode']['top_left']['y_abs']
    x_end = userconfig['processing']['geocode']['bottom_right']['x_abs']
    y_end = userconfig['processing']['geocode']['bottom_right']['y_abs']
    x_step = userconfig['processing']['geocode']['output_posting'][frequency]['x_posting']
    y_step = -1.0*userconfig['processing']['geocode']['output_posting'][frequency]['y_posting']
    epsg_code = userconfig['processing']['geocode']['outputEPSG']
    y_size = int(np.round((y_end-y_start)/y_step))
    x_size = int(np.round((x_end-x_start)/x_step))

    geo_grid = temp_isce3.pyGeoGridParameters()
    geo_grid.startX = x_start
    geo_grid.startY = y_start
    geo_grid.spacingX = x_step
    geo_grid.spacingY = y_step
    geo_grid.width = x_size
    geo_grid.length = y_size
    geo_grid.epsg = epsg_code
    
    return geo_grid
    
def _createDatasets(dst_h5, common_parent_path, frequency, polarization, shape, chunks=(128, 128)):

    dtype = np.complex64
    print("create empty dataset for frequency: {} polarization: {}".format(frequency, polarization))
    dataset_path = os.path.join(common_parent_path, f'GSLC/grids/{frequency}')
    grp = dst_h5[dataset_path]
    if chunks<shape:
        ds = grp.create_dataset(polarization, dtype=dtype, shape=shape, chunks=chunks)
    else:
        ds = grp.create_dataset(polarization, dtype=dtype, shape=shape)

    ds.attrs['description'] = np.string_(
                                      'Geocoded SLC for {} channel'.format(polarization))
    ds.attrs['units'] = np.string_('')

    return None

# end of file
