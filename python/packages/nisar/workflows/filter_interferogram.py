'''
Wrapper for interferogram filtering
'''

import pathlib
import time

from isce3.io import HDF5OptimizedReader
from isce3.signal.filter_data import filter_data, create_gaussian_kernel
import journal
import numpy as np

from nisar.workflows.filter_interferogram_runconfig import \
    FilterInterferogramRunConfig
from nisar.products.insar.product_paths import RIFGGroupsPaths
from nisar.workflows.yaml_argparse import YamlArgparse


def run(cfg: dict, input_hdf5: str):
    '''
    Run interferogram filtering
    '''
    # Pull parameters from runconfig dictionary
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    filter_args = cfg['processing']['filter_interferogram']

    # Create error and info channels
    error_channel = journal.error('filter_interferogram.run')
    info_channel = journal.info('filter_interferogram.run')
    info_channel.log("Start interferogram filtering")

    # Instantiate RIFG object to get path to RIFG datasets
    rifg_obj = RIFGGroupsPaths()

    # Check interferogram path, if not file, raise exception
    interferogram_path = filter_args['interferogram_path']
    if interferogram_path is None:
        interferogram_path = input_hdf5
    interferogram_path = pathlib.Path(interferogram_path)

    if not interferogram_path.is_file():
        err_str = f"{interferogram_path} is invalid; needs to be a file"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Record processing start time
    t_all = time.time()

    # Prepare filter kernels according to user-preference
    filter_type = filter_args['filter_type']
    if filter_type == 'no_filter':
        # No filter is required, record processing time and return
        info_channel.log('No intereferogram filtering requested')
        t_all_elapsed = time.time() - t_all
        info_channel.log(f"Ran insar filtering in {t_all_elapsed:.3f} seconds")
        return
    elif filter_type == 'boxcar':
        # Create 1D boxcar kernels in slant range/azimuth
        kernel_width = filter_args['boxcar']['filter_size_range']
        kernel_length = filter_args['boxcar']['filter_size_azimuth']
        kernel_rows = np.ones((kernel_length, 1),
                              dtype=np.float64) / kernel_length
        kernel_cols = np.ones((1, kernel_width),
                              dtype=np.float64) / kernel_width
    else:
        # Create 1D gaussian kernels centered around 0 in range/azimuth
        kernel_width = filter_args['gaussian']['filter_size_range']
        kernel_length = filter_args['gaussian']['filter_size_azimuth']
        kernel_rows = create_gaussian_kernel(kernel_length,
                                             filter_args['gaussian'][
                                                 'sigma_azimuth'])
        kernel_rows = np.reshape(kernel_rows, (kernel_length, 1))
        kernel_cols = create_gaussian_kernel(kernel_width,
                                             filter_args['gaussian'][
                                                 'sigma_range'])
        kernel_cols = np.reshape(kernel_cols, (1, kernel_width))

    # When using isce3.signal.convolve2D, it is necessary to pad the input block
    # to have an output with the same shape as the input.

    with HDF5OptimizedReader(name=input_hdf5, mode='a', libver='latest', swmr=True) as dst_h5:
        for freq, pol_list in freq_pols.items():
            freq_group_path = f'{rifg_obj.SwathsPath}/frequency{freq}'
            for pol in pol_list:

                # Get mask for that frequency/pol or general mask to be applied
                mask = get_mask(filter_args['mask'], freq, pol)

                # Get h5py igram dataset to be filtered
                pol_group_path = f'{freq_group_path}/interferogram/{pol}'
                igram_dset = dst_h5[f'{pol_group_path}/wrappedInterferogram']

                # Filter dataset
                filter_data(igram_dset, filter_args['lines_per_block'],
                       kernel_rows, kernel_cols, mask_path=mask)


    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran filter_interferogram in {t_all_elapsed:.3f} seconds")

def get_mask(mask_args, freq, pol):
    '''

    Parameters
    ----------
    mask_args: dict
        Dictionary containing mask filtering options
    freq: str
        String indicating the frequency for which extract the mask
    pol: str
        String indicating the polarization for which extract the mask

    Returns
    -------
    mask: str
        Filepath to general mask or to a mask for selected freq/pol
    '''
    mask = None

    # Check if freq and pol are in mask_args, if yes, extract mask
    if freq in mask_args:
        if pol in mask_args[freq]:
            mask = mask_args[freq][pol]
    elif 'general' in mask_args:
        mask = mask_args['general']
    return mask


if __name__ == "__main__":
    '''
    Run interferogram filtering from command line
    '''
    # Load command line args
    filter_parser = YamlArgparse()
    args = filter_parser.parse()
    # Get a runconfig dict from command line args
    filter_runconfig = FilterInterferogramRunConfig(args)
    # Use RIFG from interferogram path
    rifg_h5 = filter_runconfig.cfg['processing']['filter_interferogram'][
        'interferogram_path']
    # run insar filtering
    run(filter_runconfig.cfg, rifg_h5)
