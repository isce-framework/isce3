#!/usr/bin/env python3 #
# Author: Liang Yu
# Copyright 2019-

def runPrepHDF5(self):
    '''
    Copies shared data from RSLC HDF5 to GCOV HDF5
    '''

    import h5py
    import os
    from nisar.h5 import cp_h5_meta_data

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
                    os.path.join(common_parent_path, 'GCOV/metadata/orbit'))
    cp_h5_meta_data(src_h5, dst_h5, 
                    os.path.join(common_parent_path, 'SLC/metadata/attitude'),
                    os.path.join(common_parent_path, 'GCOV/metadata/attitude'))

    # copy calibration information group
    cp_h5_meta_data(src_h5, dst_h5,
            os.path.join(common_parent_path, 'SLC/metadata/calibrationInformation'),
            os.path.join(common_parent_path, 'GCOV/metadata/calibrationInformation'),
            excludes=['zeroDopplerTime', 'slantRange', 'geometry'])

    # copy processing information group
    cp_h5_meta_data(src_h5, dst_h5,
            os.path.join(common_parent_path, 'SLC/metadata/processingInformation'),
            os.path.join(common_parent_path, 'GCOV/metadata/processingInformation'),
            excludes=['l0bGranules', 'demFiles', 'zeroDopplerTime', 'slantRange'])

    # copy radar grid information group
    cp_h5_meta_data(src_h5, dst_h5,
            os.path.join(common_parent_path, 'SLC/metadata/geolocationGrid'),
            os.path.join(common_parent_path, 'GCOV/metadata/radarGrid'),
            renames={'coordinateX':'xCoordinates',
                'coordinateY':'yCoordinates',
                'zeroDopplerTime':'zeroDopplerAzimuthTime'})

    # copy radar imagery group; assumming shared data
    # XXX option0: to be replaced with actual gcov code
    # XXX option1: do not write GCOV data here; GCOV rasters can be appended to the GCOV HDF5
    for freq in ['A', 'B']:
        ds_ref = os.path.join(common_parent_path, f'SLC/swaths/frequency{freq}')
        if ds_ref not in src_h5:
            continue
        cp_h5_meta_data(src_h5, dst_h5,
                ds_ref,
                os.path.join(common_parent_path, f'GCOV/grids/frequency{freq}'),
                excludes=['acquiredCenterFrequency', 'acquiredAzimuthBandwidth', 
                    'acquiredRangeBandwidth', 'nominalAcquisitionPRF', 'slantRange',
                    'sceneCenterGroundRangeSpacing',
                    'HH', 'HV', 'VH', 'VV', 'RH', 'RV',
                    'validSamplesSubSwath1', 'validSamplesSubSwath2',
                    'validSamplesSubSwath3', 'validSamplesSubSwath4'],
                renames={'processedCenterFrequency':'centerFrequency',
                         'processedAzimuthBandwidth':'azimuthBandwidth',
                         'processedRangeBandwidth':'rangeBandwidth'})

# end of file
