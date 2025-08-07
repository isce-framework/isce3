#!/usr/bin/env python3 #

def cp_h5_meta_data(src_h5, dst_h5, src_path, dst_path=None,
        excludes=None, renames=None, flag_overwrite=False,
        attach_scales_list = None): 
    '''
    Copy HDF5 node contents

    Parameters:
    -----------
    src_h5 : h5py object
        h5py object from source HDF5 file
    dst_h5 : h5py object
        h5py object from destination HDF5 file
    src_path : str
        Full node path in source HDF5 to be copied
    dst_path : str, optional
        Full node path in destination HDF5 to be copied to
        Is set to src_path if value not provided
    excludes : list of str, optional
        Names of nodes to skip in src_path
    renames : dict of (str, str), optional
        Dict where keys are source node names and vales
        are destination node names
    flag_overwrite :  bool, optional
        If flag_overwrite, delete existing dataset
    attach_scales_list : list, optional
        List of new dimensions (scales) to overwrite
        existing dimensions attributes in destination
        datasets
    '''

    import h5py
    import os

    # assign defaults as needed
    if dst_path is None:
        dst_path = src_path
    if excludes is None:
        excludes = []
    if renames is None:
        renames = {}

    # get src group at src_path
    src_group = src_h5[src_path]

    # get dst parent path and create if needed
    dst_parent_path, dst_name = os.path.split(dst_path)
    if dst_parent_path not in dst_h5:
        dst_h5.create_group(dst_parent_path)

    if excludes or renames or attach_scales_list is not None:

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

            # if flag_overwrite, delete existing dataset
            if flag_overwrite and subnode_dst in dst_group:
                del dst_group[subnode_dst]

            # copy group/dataset
            dst_group.copy(node_obj, subnode_dst)
            if (attach_scales_list is not None and 
                    'DIMENSION_LIST' in node_obj.attrs.keys()):
                dst_group[subnode_dst].attrs.__delitem__('DIMENSION_LIST')
                for i, dim in enumerate(attach_scales_list):
                    dst_group[subnode_dst].dims[i].attach_scale(dim)
    else:
        dst_group = dst_h5[dst_parent_path]

        # if flag_overwrite, delete existing dataset
        if flag_overwrite and dst_name in dst_group:
            del dst_group[dst_name]

        # simple group copy of todos
        dst_group.copy(src_group, dst_name)


# end of file
