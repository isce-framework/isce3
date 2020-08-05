#!/usr/bin/env python3 #
# Author: Liang Yu
# Copyright 2019-

def cp_h5_meta_data(src_h5, dst_h5, src_path, dst_path='',
        excludes=None, renames=None): 
    '''
    Copy HDF5 node contents

    Parameters:
    -----------
    src_h5 : h5py object
        h5py object from source HDF5 file
    dst_h5 : h5py object
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

    import h5py
    import os

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
            if excludes is not None and subnode_src in excludes:
                continue
            subnode_dst = subnode_src
            if renames is not None and subnode_src in renames:
                subnode_dst = renames[subnode_src]

            src_sub_path = os.path.join(src_path, subnode_src)
            node_obj = src_h5[src_sub_path]

            # copy group/dataset
            dst_group.copy(node_obj, subnode_dst)
    else:
        # simple group copy of todos
        dst_h5[dst_parent_path].copy(src_group, dst_name)


# end of file
