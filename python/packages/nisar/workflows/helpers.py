'''
collection of useful functions used across workflows
'''

from collections import defaultdict
import os
import pathlib

from osgeo import gdal
import h5py
import nisar
import numpy as np

import journal

from nisar.products.readers import SLC

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


def check_mode_directory_tree(parent_dir: str, mode: str, frequency_list: list, pols: dict = {}):
    '''
    Checks existence parent directory and sub-directories.
    Sub-directories made from mode sub_dir + frequency_list.
    Expected directory tree:
    outdir/
    └── mode/
        └── freq(A,B)
            └── (HH, HV, VH, VV)
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
        raise NotADirectoryError(err_str)

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
        if not freq_dir.is_dir():
            err_str = f"{str(freq_dir)} not a valid path"
            error_channel.log(err_str)
            raise NotADirectoryError(err_str)

        # if no polarizations given continue to check next frequency
        if not pols:
            continue

        # check if frequency in polarization dict
        if freq not in pols:
            err_str = f"No key in polarization dict for frequency: {freq}"
            error_channel.log(err_str)
            raise KeyError(err_str)

        # check if polarization directory exists
        for pol in pols[freq]:
            pol_dir = freq_dir / pol
            if not pol_dir.is_dir():
                err_str = f"{str(pol_dir)} not a valid path"
                error_channel.log(err_str)
                raise NotADirectoryError(err_str)


def check_hdf5_freq_pols(h5_path: str, freq_pols: dict):
    '''
    Check if frequency (keys) and polarizations (items) exist in HDF5
    Expected HDF5 structure:
    swath or grid group/
    └── freq(A,B) group
        └── (HH, HV, VH, VV) dataset
    '''
    error_channel = journal.error('helpers.check_hdf5_freq_pols')

    # attempt to open HDF5
    try:
        h5_obj = h5py.File(h5_path, 'r', libver='latest', swmr=True)
    except:
        err_str = f"h5py unable to open {h5_path}"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # use with to ensure h5_obj closes
    with h5_obj:
        product_type = h5_obj['/science/LSAR/identification/productType'][()].decode('UTF-8')
        if product_type.startswith('G'):
            grid_type = 'grids'
        else:
            grid_type = 'swaths'
        grid_path = f'/science/LSAR/{product_type}/{grid_type}'

        # get swath/grid group from hdf5
        grid_group = h5_obj[grid_path]

        # check if frequencies in group
        for freq in freq_pols:
            freq_str = f"frequency{freq}"
            if freq_str not in grid_group:
                err_str = f"{freq} not found in swath/grid group of {h5_path}"
                error_channel.log(err_str)
                raise ValueError(err_str)

            # get frequency group from swath/grid group
            freq_group = grid_group[freq_str]
            if 'interferogram' in freq_group:
                freq_group = freq_group['interferogram']

            # check if polarizations in group
            for pol in freq_pols[freq]:
                if pol not in freq_group:
                    err_str = f"{pol} not found in {freq} group of swath/grid group of {h5_path}"
                    error_channel.log(err_str)
                    raise ValueError(err_str)


def copy_raster(infile, freq, pol,
                lines_per_block, outfile, file_type="ENVI"):
    '''
    Copy RSLC dataset to GDAL format and convert real and
    imaginary parts from float16 to float32

    Parameters
    ----------
    infile: str
        Path to RSLC HDF5
    freq: str
        RSLC frequency band to process ('A' or 'B')
    pol: str
        RSLC polarization to process
    outfile: str
        Output filename
    file_type: str
        GDAL-friendly file format
    '''

    # Open RSLC HDF5 file dataset
    rslc = SLC(hdf5file=infile)
    hdf5_ds = rslc.getSlcDataset(freq, pol)

    # Get RSLC dimension through GDAL
    gdal_ds = gdal.Open(f'HDF5:{infile}:/{rslc.slcPath(freq, pol)}')
    rslc_length, rslc_width = gdal_ds.RasterYSize, gdal_ds.RasterXSize

    # Create output file
    driver = gdal.GetDriverByName(file_type)
    out_ds = driver.Create(outfile, rslc_width, rslc_length,
                           1, gdal.GDT_CFloat32)

    # Start block processing
    lines_per_block = min(rslc_length, lines_per_block)
    num_blocks = int(np.ceil(rslc_length / lines_per_block))

    # Iterate over blocks to convert and write
    for block in range(num_blocks):
        line_start = block * lines_per_block

        # Check for last block and compute block length accordingly
        if block == num_blocks - 1:
            block_length = rslc_length - line_start
        else:
            block_length = lines_per_block

        # Read a block of data from RSLC and convert real and imag part to float32
        s = np.s_[line_start:line_start + block_length, :]
        data_block = nisar.types.read_c4_dataset_as_c8(hdf5_ds, s)

        # Write to GDAL raster
        out_ds.GetRasterBand(1).WriteArray(data_block[0:block_length],
                                           yoff=line_start, xoff=0)
    out_ds.FlushCache()


def complex_raster_path_from_h5(slc, freq, pol, hdf5_path, lines_per_block,
                                c32_output_path):
    '''
    Get path for io.raster based on raster datatype. If datatype is not
    complex64,convert and save to temporary file. Raster object generated here
    to avoid potential artifacts caused by copying for Raster objects.

    Parameters
    ----------
    slc: nisar.products.readers.SLC
        RSLC object
    freq: str
        RSLC frequency band to process ('A' or 'B')
    pol: str
        RSLC polarization to process
    hdf5_path: str
        Source HDF5 file
    lines_per_block: int
        Lines per block to be converted and written to complex32 (if needed)
    c32_output_path: str
        GDAL-friendly file format

    Returns
    -------
    raster_path: str
        isce3.io.Raster-friendly path to raster dataset
    file_path: str
        File containing raster dataset. Differs from raster_path if when output
        is HDF5
    '''
    if slc.is_dataset_complex64(freq, pol):
        # If SLC dataset is complex64 HDF5, return GDAL path to HDF5 dataset
        slc_h5_path = f'/{slc.SwathPath}/frequency{freq}/{pol}'
        raster_path = f'HDF5:{hdf5_path}:{slc_h5_path}'
        file_path = hdf5_path
    else:
        # If SLC dataset is not complex64 HDF5, covert to complex32, write to
        # ENVI raster, and return path ENVI raster
        copy_raster(hdf5_path, freq, pol, lines_per_block,
                    c32_output_path, file_type='ENVI')
        raster_path = c32_output_path
        file_path = c32_output_path

    return raster_path, file_path
