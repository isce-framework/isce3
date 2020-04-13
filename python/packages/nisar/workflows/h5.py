import h5py
import os

def cp_h5_meta_data(src_h5, dst_h5, src_path, dst_path='',
        excludes=[], renames={}): 
    '''
    Copy HDF5 node contents

    Parameters:
    -----------
    src_h5 : str
        h5py object from source HDF5 file
    dst_h5 : str
        h5py object from destination HDF5 file
    src_path : str
        Full path in source HDF5 to be copied
    dst_path : str, optional
        Full path in destination HDF5 to be copied to
        Is set to src_path if value not provided
    excludes : list of str, optional
        Names of nodes to skip in src_path
    renames : dict of (str, str), optional
        Dict where keys are source node names and vales
        are destination node names
    '''

    # get src group at src_path
    src_group = src_h5[src_path]

    # reset dst path if needed
    if not dst_path:
        dst_path = src_path

    # get dst parent path and create if needed
    dst_parent_path, dst_name = os.path.split(dst_path)
    if dst_parent_path not in dst_h5:
        dst_h5.create_group(dst_parent_path)

    if excludes or renames:
        # copy dataset piecemeal from src_h5:src_path to dst_h5:dst_path
        # create dst_path if needed
        if dst_path not in dst_h5:
            dst_h5.create_group(dst_path)
        dst_group = dst_h5[dst_path]

        for subnode_src in src_group.keys(): 
            # check conditions while copying piecemeal
            if subnode_src in excludes:
                continue
            subnode_dst = subnode_src
            if subnode_src in renames:
                subnode_dst = renames[subnode_src]

            src_sub_path = os.path.join(src_path, subnode_src)
            node_obj = src_h5[src_sub_path]

            if type(node_obj) == h5py._hl.group.Group:
                # copy group
                dst_group.copy(node_obj, subnode_dst)
            elif type(node_obj) == h5py._hl.dataset.Dataset:
                # copy dataset
                dst_group.create_dataset(subnode_dst, data=node_obj)
    else:
        # simple group copy of todos
        dst_h5[dst_parent_path].copy(src_group, dst_name)


def prep_gcov_h5(path_src, path_dst):
    '''
    Copies shared data from RSLC HDF5 to GCOV HDF5

    Parameters:
    -----------
    path_src : str
        Full path to source HDF5 file
    path_dst : str
        Full path to destination HDF5 file
    '''
    # prelim setup
    common_parent_path = 'science/LSAR'
    src_h5 = h5py.File(path_src, 'r')

    # rm anything and start from scratch
    try:
        os.remove(path_dst)
    except FileNotFoundError:
        pass

    dst_h5 = h5py.File(path_dst, 'w')

    # simple copies of identification, metadata/orbit, metadata/attitude groups
    cp_h5_meta_data(src_h5, dst_h5, os.path.join(common_parent_path, 'identification'))
    cp_h5_meta_data(src_h5, dst_h5, os.path.join(common_parent_path, 'SLC/metadata/orbit'))
    cp_h5_meta_data(src_h5, dst_h5, os.path.join(common_parent_path, 'SLC/metadata/attitude'))

    # copy calibration information group
    cp_h5_meta_data(src_h5, dst_h5,
            os.path.join(common_parent_path, 'SLC/metadata/calibrationInformation'),
            os.path.join(common_parent_path, 'GCOV/metadata/calibrationInformation'),
            excludes=['zeroDopplerTime', 'slantRange'])
                
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
        cp_h5_meta_data(src_h5, dst_h5,
                os.path.join(common_parent_path, f'SLC/swaths/frequency{freq}'),
                os.path.join(common_parent_path, f'GCOV/grids/frequency{freq}'),
                excludes=['acquiredCenterFreqeuncy', 'acquiredAzimuthBandwidth', 
                    'acquiredRangeBandwidth', 'nominalAcquisitionPRF', 'slantRange',
                    'sceneCenterAlongTrackSpacing', 'sceneCenterGroundRangeSpacing',
                    'HH', 'HV', 'VH', 'VV', 'RH', 'RV',
                    'validSamplesSubSwath1', 'validSamplesSubSwath2',
                    'validSamplesSubSwath3', 'validSamplesSubSwath4'],
                renames={'processedCenterFrequency':'centerFrequency',
                    'processedAzimuthBandwidth':'azimuthBandwidth',
                    'processedRangeBandwidth':'rangeBandwidth'})


if __name__ == '__main__':
    import sys

    path_src = sys.argv[1]
    path_dst = sys.argv[2]
    
    prep_gcov_h5(path_src, path_dst)

# end of file
