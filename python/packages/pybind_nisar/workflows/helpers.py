'''
collection of useful functions used across workflows
'''

from collections import defaultdict
import os
import pathlib

import gdal

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
    if not dst_path:
        dst_path = '.'

    error_channel = journal.error('helpers.check_write_dir')

    # check if scratch path exists
    dst_path_ok = os.path.isdir(dst_path)

    if not dst_path_ok:
        try:
            os.makedirs(dst_path, exist_ok=True)
        except OSError:
            err_str = f"Unable to create {dst_path}"
            error_channel.log(err_str)
            raise OSError(err_str)

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

    try:
        gdal.Open(dem_path)
    except:
        err_str = f'{dem_path} cannot be opened by GDAL'
        error_channel.log(err_str)
        raise ValueError(err_str)


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


def check_mode_directory_tree(parent_dir: str, mode: str, frequency_list: list):
    '''
    Checks existence parent directory and sub-directories.
    Sub-directories made from mode sub_dir + frequency_list.
    Expected directory tree:
    outdir/
    └── mode/
       ├── freqA/
       └── freqB/
    '''
    error_channel = journal.error('helpers.check_directory_tree')

    parent_dir = pathlib.Path(parent_dir)

    # check if parent is a directory
    if not parent_dir.is_dir():
        err_str = f"{str(parent_dir)} not a valid path"
        error_channel.log(err_str)
        raise NotADirectoryError(err_str)

    # check if mode-directory exists
    mode_dir = parent_dir / f'{mode}'
    if not mode_dir.is_dir():
        err_str = f"{str(mode_dir)} not a valid path"
        error_channel.log(err_str)

    # check number frequencies
    n_frequencies = len(frequency_list)
    if n_frequencies not in [1, 2]:
        err_str = f"{n_frequencies} is an invalid number of frequencies. Only 1 or 2 frequencies allowed"
        error_channel.log(err_str)
        raise ValueError(err_str)

    for freq in frequency_list:
        # check if frequency allowed
        if freq not in ['A', 'B']:
            err_str = f"frequency {freq} not valid. Only [A, B] allowed."
            error_channel.log(err_str)
            raise ValueError(err_str)

        # check if mode-directory exists
        freq_dir = mode_dir / f'freq{freq}'
        if not mode_dir.is_dir():
            err_str = f"{str(freq_dir)} not a valid path"
            error_channel.log(err_str)
            raise NotADirectoryError(err_str)
