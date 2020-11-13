'''
collection of useful functions used across workflows
'''

from collections import defaultdict
import os

import journal


def deep_update(original, update):
    '''
    update default runconfig key with user supplied dict
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    '''
    for key, val in update.items():
        if isinstance(val, dict):
            original[key] = deep_update(original.get(key, {}), val)
        else:
            original[key] = val

    # return updated original
    return original


def autovivified_dict():
    '''
    Use autovivification to create nested dictionaries.
    https://en.wikipedia.org/wiki/Autovivification

    defaultdict creates any items you try to access if they don't exist yet.
    defaultdict only performs this for a single level.
    https://stackoverflow.com/a/5900634

    The recursion extends this behavior and allows the creation of additional levels.
    https://stackoverflow.com/a/22455426

    '''
    return defaultdict(autovivified_dict)


WORKFLOW_SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))


def check_write_dir(dst_path: str):
    '''
    Raise error if given path does not exist or not writeable.
    '''
    error_channel = journal.error('helpers.check_check_write_dir')

    # check if scratch path exists
    dst_path_ok = os.path.isdir(dst_path)
    if not dst_path_ok:
        err_str = f"{dst_path} scratch directory does not exist."
        error_channel.log(err_str)
        raise NotADirectoryError(err_str)

    # check if path writeable
    write_ok = os.access(dst_path, os.W_OK)
    if not write_ok:
        err_str = f"{dst_path} scratch directory lacks write permission."
        error_channel.log(err_str)
        raise PermissionError(err_str)


def check_dem(dem_path: str):
    '''
    Raise error if DEM is not system file, netCDF, nor S3.
    '''
    error_channel = journal.error('helpers.check_dem')

    if dem_path.startswith('/vsi3') or dem_path.startswith('NETCDF:'):
        try:
            gdal.Open(dem_path)
        except:
            err_str = f'{dem_path} cannot be opened by GDAL'
            error_channel.log(err_str)
            raise ValueError(err_str)

    if not os.path.isfile(dem_path):
        err_str = f"{dem_path} not valid"
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)


def check_log_dir_writable(log_file_path: str):
    '''
    Check to see if destination directory of log file path is writable.
    Raise error if directory lacks write permission.
    '''
    error_channel = journal.error('helpers.check_log_dir_writeable')

    dest_dir, _ = os.path.split(log_file_path)

    # get current working directory if no directory in run_config_path
    if not dest_dir:
        dest_dir = os.getcwd()

    if not os.access(dest_dir, os.W_OK):
        err_str = f"No write permission to {dest_dir}"
        error_channel.log(err_str)
        raise PermissionError(err_str)
