#!/usr/bin/env python3 #
# Author: Liang Yu
# Copyright 2019-

import os
import numpy as np

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

    ident = dst_h5[os.path.join(common_parent_path, 'identification')]
    dset = ident.create_dataset('isGeocoded', data=np.string_("True"))
    desc = f"Flag to indicate radar geometry or geocoded product"
    dset.attrs["description"] = np.string_(desc)

    cp_h5_meta_data(src_h5, dst_h5, 
                    os.path.join(common_parent_path, 'SLC/metadata/orbit'),
                    os.path.join(common_parent_path, 'GCOV/metadata/orbit'))
    cp_h5_meta_data(src_h5, dst_h5, 
                    os.path.join(common_parent_path, 'SLC/metadata/attitude'),
                    os.path.join(common_parent_path, 'GCOV/metadata/attitude'))
    
    # copy calibration information group
    for freq in state.subset_dict.keys():
          frequency = f'frequency{freq}'
          pol_list = state.subset_dict[freq]
          for polarization in pol_list:
              cp_h5_meta_data(src_h5, dst_h5,
              os.path.join(common_parent_path, 
                  f'SLC/metadata/calibrationInformation/{frequency}/{polarization}'),
              os.path.join(common_parent_path, 
                  f'GCOV/metadata/calibrationInformation/{frequency}/{polarization}'))

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

    input_grp = dst_h5[os.path.join(common_parent_path, 
                       'GCOV/metadata/processingInformation/inputs')]
    dset = input_grp.create_dataset("l1SlcGranules", data=np.string_([state.input_hdf5]))
    desc = f"List of input L1 products used"
    dset.attrs["description"] = np.string_(desc)

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
                    'validSamplesSubSwath3', 'validSamplesSubSwath4',
                    'listOfPolarizations'],
                renames={'processedCenterFrequency':'centerFrequency',
                         'processedAzimuthBandwidth':'azimuthBandwidth',
                         'processedRangeBandwidth':'rangeBandwidth'})
    for freq in state.subset_dict.keys():
        frequency = f'frequency{freq}'
        pol_list = state.subset_dict[freq]
        _addPolarizationList(dst_h5, common_parent_path, frequency, pol_list)


def _addPolarizationList(dst_h5, common_parent_path, frequency, pols):

    for pol in pols:
        assert len(pol) == 2 and pol[0] in "HVLR" and pol[1] in "HV"
    dataset_path = os.path.join(common_parent_path, f'GCOV/grids/{frequency}')
    grp = dst_h5[dataset_path]
    name = "listOfPolarizations"
    polsArray = np.array(pols, dtype="S2")
    dset = grp.create_dataset(name, data=polsArray)
    desc = f"List of polarization layers with frequency{frequency}"
    dset.attrs["description"] = np.string_(desc)

    return None

# end of file
