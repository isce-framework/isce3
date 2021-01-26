'''
wrapper for crossmul
'''
import pathlib
import time

import h5py
import journal

import pybind_isce3 as isce3
from pybind_nisar.products.readers import SLC
from pybind_nisar.workflows import h5_prep
from pybind_nisar.workflows.crossmul_argparse import CrossmulArgparse
from pybind_nisar.workflows.crossmul_runconfig import CrossmulRunConfig

def run(cfg: dict, output_hdf5: str = None):
    '''
    run crossmul
    '''
    # pull parameters from cfg
    ref_hdf5 = cfg['InputFileGroup']['InputFilePath']
    sec_hdf5 = cfg['InputFileGroup']['SecondaryFilePath']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    flatten = cfg['processing']['crossmul']['flatten']

    if flatten is not None:
        flatten_path = cfg['processing']['crossmul']['flatten']

    if output_hdf5 is None:
        output_hdf5 = cfg['ProductPathGroup']['SASOutputFile']

    # init parameters shared by frequency A and B
    ref_slc = SLC(hdf5file=ref_hdf5)
    sec_slc = SLC(hdf5file=sec_hdf5)

    error_channel = journal.error('crossmul.run')
    info_channel = journal.info("crossmul.run")
    info_channel.log("starting crossmultipy")

    # for now only use CPU
    crossmul = isce3.signal.Crossmul()

    crossmul.range_looks = cfg['processing']['crossmul']['range_looks']
    crossmul.az_looks = cfg['processing']['crossmul']['azimuth_looks']
    crossmul.oversample = cfg['processing']['crossmul']['oversample']

    # check if user provided path to raster(s) is a file or directory
    coregistered_slc_path = pathlib.Path(cfg['processing']['crossmul']['coregistered_slc_path'])
    coregistered_is_file = coregistered_slc_path.is_file()
    if not coregistered_is_file and not coregistered_slc_path.is_dir():
        err_str = f"{coregistered_slc_path} is invalid; needs to be a file or directory."
        error_channel.log(err_str)
        raise ValueError(err_str)

    t_all = time.time()
    with h5py.File(output_hdf5, 'a', libver='latest', swmr=True) as dst_h5:
        for freq, pol_list in freq_pols.items():
            # get 2d doppler, discard azimuth dependency, and set crossmul dopplers
            ref_dopp = isce3.core.LUT1d(ref_slc.getDopplerCentroid(frequency=freq))
            sec_dopp = isce3.core.LUT1d(sec_slc.getDopplerCentroid(frequency=freq))
            crossmul.set_dopplers(ref_dopp, sec_dopp)

            freq_group_path = f'/science/LSAR/RIFG/swaths/frequency{freq}'

            if flatten is not None:
                # set frequency dependent range offset raster
                flatten_raster = isce3.io.Raster(f'{flatten_path}/geo2rdr/freq{freq}/range.off')

                # prepare range filter parameters
                rdr_grid = ref_slc.getRadarGrid(freq)
                rg_pxl_spacing = rdr_grid.range_pixel_spacing
                wavelength = rdr_grid.wavelength
                rg_sample_freq = isce3.core.speed_of_light / 2.0 / rg_pxl_spacing
                rg_bandwidth = ref_slc.getSwathMetadata(freq).processed_range_bandwidth

                # set crossmul range filter
                crossmul.set_rg_filter(rg_sample_freq, rg_bandwidth, rg_pxl_spacing, wavelength)

            for pol in pol_list:
                pol_group_path = f'{freq_group_path}/interferogram/{pol}'

                # prepare reference input raster
                ref_raster_str = f'HDF5:{ref_hdf5}:/{ref_slc.slcPath(freq, pol)}'
                ref_slc_raster = isce3.io.Raster(ref_raster_str)

                # prepare secondary input raster
                if coregistered_is_file:
                    raster_str = f'HDF5:{sec_hdf5}:/{sec_slc.slcPath(freq, pol)}'
                else:
                    raster_str = str(coregistered_slc_path /\
                                     f'resample_slc/freq{freq}/{pol}/coregistered_secondary.slc')

                sec_slc_raster = isce3.io.Raster(raster_str)

                # access the HDF5 dataset for a given frequency and polarization
                dataset_path = f'{pol_group_path}/wrappedInterferogram'
                igram_dataset = dst_h5[dataset_path]

                # Construct the output ratster directly from HDF5 dataset
                igram_raster = isce3.io.Raster(f"IH5:::ID={igram_dataset.id.id}".encode("utf-8"),
                                               update=True)

                # call crossmul with coherence if multilooked
                if crossmul.range_looks > 1 or crossmul.az_looks > 1:
                    # access the HDF5 dataset for a given frequency and polarization
                    dataset_path = f'{pol_group_path}/coherenceMagnitude'
                    coherence_dataset = dst_h5[dataset_path]

                    # Construct the output ratster directly from HDF5 dataset
                    coherence_raster = isce3.io.Raster(
                            f"IH5:::ID={coherence_dataset.id.id}".encode("utf-8"), update=True)

                    if flatten is not None:
                        crossmul.crossmul(ref_slc_raster, sec_slc_raster, flatten_raster,
                                          igram_raster, coherence_raster)
                    else:
                        crossmul.crossmul(ref_slc_raster, sec_slc_raster,
                                          igram_raster, coherence_raster)

                    del coherence_raster
                else:
                    # no coherence without multilook
                    crossmul.crossmul(ref_slc_raster, sec_slc_raster, igram_raster)

                del igram_raster

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran crossmul in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    '''
    run crossmul from command line
    '''
    # load command line args
    crossmul_parser = CrossmulArgparse()
    args = crossmul_parser.parse()
    # get a runconfig dict from command line args
    crossmul_runconfig = CrossmulRunConfig(args)
    # prepare RIFG HDF5
    out_paths = h5_prep.run(crossmul_runconfig.cfg)
    # run crossmul
    run(crossmul_runconfig.cfg, out_paths['RIFG'])
