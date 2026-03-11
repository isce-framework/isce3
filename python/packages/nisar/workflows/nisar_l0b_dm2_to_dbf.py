#!/usr/bin/env python3
"""
Produce NISAR DBF/Science-like L0B product from NISAR DM2 L0B
"""
import argparse
from pathlib import Path
import os
import time
import tempfile
from copy import copy
from datetime import datetime

import numpy as np
from scipy.signal import fftconvolve
import h5py
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from nisar.products.readers.Raw import Raw
from nisar.products.readers.antenna import AntennaParser
from nisar.products.readers import (
    load_attitude_from_xml, load_orbit_from_xml)
from isce3.geometry import DEMInterpolator
from isce3.io import Raster
from nisar.log import set_logger
from isce3.signal import dbf_onetap_from_dm2, dbf_onetap_from_dm2_seamless
from isce3.focus import fill_gaps, form_linear_chirp
from isce3.core import TimeDelta
from nisar.workflows.helpers import build_uniform_quantizer_lut_l0b, slice_gen
from isce3.antenna import ant2rgdop


def _join_paths(path1: str, path2: str) -> str:
    """Join two paths to be used in HDF5"""
    sep = '/'
    if path1.endswith(sep):
        sep = ''
    return path1 + sep + path2


def copy_swath_except_echo_h5(
        fid_in, fid_out, swath_path, frq_pol, tx_keys=None, rx_keys=None):
    """
    Copy all groups and datasets under swath from input HDF5 to output
    HDF5 except for echo and optionally for desired datasets under tx and
    rx group listed by tx_keys and rx_keys.
    "validSamplesSubSwath*" on tx group will be excluded from being copied.

    Parameters
    ----------
    fid_in : h5py.File
        File-like object for input HDF5 L0B product
    fid_out : h5py.File
        File-like object for output HDF5 L0B product
    swath_path : str
        HDF5 path for swath.
    frq_pol : dict
        A dict of all frequency bands as keys and list of TxRx
        polarization as values.
    tx_keys : sequence of str, optional
        Sequence of dataset names to be excluded from tx group in `fid_out`.
    rx_keys : sequence of str, optional
        Sequence of dataset names to be excluded from rx group in `fid_out`.
        Echo datasets are always excluded from RX group!

    """
    # TX/RX dataset names to be excluded over all frequency bands and pols!
    tx_names = []
    if tx_keys is not None:
        tx_names.extend(tx_keys)
    rx_names = []
    if rx_keys is not None:
        rx_names.extend(rx_keys)

    for freq_band in frq_pol:
        # build band path
        band_path = _join_paths(swath_path, f'frequency{freq_band}/')
        # create freq band group for output product
        grp_band = fid_out.require_group(band_path)
        # list of all TxRx products
        txrx_pols = frq_pol[freq_band]
        # form rx datset names plus echo for all pols to be
        # excluded from output
        rx_names_echo = copy(txrx_pols)
        rx_names_echo.extend(rx_names)
        # list of group names txp where "p" is all TX pols
        txp = {f'tx{p[0]}' for p in txrx_pols}
        # list of group names rxp where "p" is all RX pols
        rxp = {f'rx{p[1]}' for p in txrx_pols}
        for band_item in fid_in[band_path]:
            if band_item in txp:
                # form TX path
                tx_path = band_path + band_item + '/'
                # create TX group for output product
                grp_tx = fid_out.require_group(tx_path)
                for tx_item in fid_in[tx_path]:
                    if tx_item in rxp:
                        # form RX path
                        rx_path = tx_path + tx_item + '/'
                        # create RX group for output product
                        grp_rx = fid_out.require_group(rx_path)
                        for rx_item in fid_in[rx_path]:
                            if rx_item not in rx_names_echo:
                                # copy all RX fields except
                                # echo + rx_keys
                                fid_in.copy(rx_path + rx_item, grp_rx)
                    # copy the rest of TX that is not in tx_keys
                    elif ((tx_item not in tx_names) and
                          ('validSamplesSubSwath' not in tx_item)):
                        fid_in.copy(tx_path + tx_item, grp_tx)

            else:
                # copy the rest of frequency band as it is
                fid_in.copy(band_path + band_item, grp_band)


def create_echo_dataset_h5(fid_in, fid_out, band_path, txrx_pol,
                           prod_shape, prod_dtype, comp_level_h5):
    """
    Create a new echo dataset in output HDF5 w/ the same attribute as
    that of input HDF5 per desired shape and data type.

    Parameters
    ----------
    fid_in : h5py.File
        File-like object for input HDF5 L0B product
    fid_out : h5py.File
        File-like object for output HDF5 L0B product
    band_path : str
        Frequency band path in L0B HDF5 product.
    txrx_pol : str
        TxRx polarization of the new echo product
    prod_shape: tuple(int, int)
        Shape of the new echo product
    prod_dtype: np.dtype
        Date type of the new echo product
    comp_level_h5 : int
        Compression level of gzip in HDF5 used for echo ratser.

    Returns
    -------
    h5py.Dateset
        HDF5 dataset for the output echo product

    Notes
    -----
    It is assumed the parent group containing datasets has already existed
    in output HDF5

    """
    rx_path = _join_paths(band_path, f'tx{txrx_pol[0]}/rx{txrx_pol[1]}/')
    # create a place holder for dataset
    grp_rx = fid_out[rx_path]
    dset_prod = grp_rx.create_dataset(
        txrx_pol, shape=prod_shape, dtype=prod_dtype,
        chunks=True, compression='gzip',
        compression_opts=comp_level_h5
    )
    # copy attributes for the product from input
    p_path = rx_path + txrx_pol
    for a_name, a_val in fid_in[p_path].attrs.items():
        dset_prod.attrs[a_name] = a_val
    return dset_prod


def _copy_datasets_truncated(
        fid_in, fid_out, band_path, pol, pulse_slice, logger, keys):
    """
    Copy truncated version of non-echo pulse-dependent datasets under
    either tx or rx group from input HDF5 to output one given desired
    input pulse slice.
    The type of tx or rx is determined by the size of pol.
    If single char then it is tx group and if two-char str then
    it will be dataset under rx group

    Paramaters
    ----------
    fid_in : h5py.File
        File-like object for input HDF5 L0B product
    fid_out : h5py.File
        File-like object for output HDF5 L0B product
    band_path : str
        Frequency band path in L0B HDF5 product.
    pol : str
        Either signle-char tx pol or two-char TxRx polarization
        of the new echo product
    pulse_slice : slice
        Desired pulse slice for input HDF5 file.
    logger: logging.Logger
    keys: sequence of str
        Desired keys under RX group to be copied as long as
        they exist in input hdf5.

    """
    path = _join_paths(band_path, f'tx{pol[0]}/')
    if len(pol) == 2:
        path += f'rx{pol[1]}/'
    # output group
    grp_out = fid_out[path]
    for name in set(keys):
        try:
            dset_in = fid_in[path + name]
        except KeyError as err:
            logger.warning(
                f'Missing dataset "{path + name}" in input L0B!'
                f' Detailed Error -> "{err}"'
            )
            continue
        else:
            data_in = dset_in[pulse_slice]
            dset_out = grp_out.require_dataset(
                name, shape=data_in.shape, dtype=data_in.dtype, data=data_in)
            # copy attributes for the product from input
            for a_name, a_val in dset_in.attrs.items():
                dset_out.attrs[a_name] = a_val


def copy_rx_datasets_truncated(
        fid_in, fid_out, band_path, txrx_pol, pulse_slice, logger,
        keys=('caltone', 'RD', 'WD', 'WL', 'basebandPhaseCorrection',
              'TRMDataWindow', 'attenuation')
):
    """
    Copy truncated version of non-echo pulse-dependent datasets under rx group
    from input HDF5 to output one given desired input pulse slice.

    Paramaters
    ----------
    fid_in : h5py.File
        File-like object for input HDF5 L0B product
    fid_out : h5py.File
        File-like object for output HDF5 L0B product
    band_path : str
        Frequency band path in L0B HDF5 product.
    txrx_pol : str
        TxRx polarization of the new echo product
    pulse_slice : slice
        Desired pulse slice for input HDF5 file.
    logger: logging.Logger
    keys: sequence of str
        Desired keys under RX group to be copied as long as
        they exist in input hdf5.
        Default is ('caltone', 'RD', 'WD', 'WL', 'basebandPhaseCorrection',
        'TRMDataWindow', 'attenuation').

    """
    _copy_datasets_truncated(
        fid_in, fid_out, band_path, txrx_pol, pulse_slice, logger, keys)


def copy_tx_datasets_truncated(
        fid_in, fid_out, band_path, tx_pol, pulse_slice, logger,
        keys=('rangeLineIndex', 'radarTime', 'UTCtime', 'txPhase',
              'calType', 'chirpCorrelator')
):
    """
    Copy truncated version of non-echo pulse-dependent datasets under tx group
    from input HDF5 to output one given desired input pulse slice.

    Paramaters
    ----------
    fid_in : h5py.File
        File-like object for input HDF5 L0B product
    fid_out : h5py.File
        File-like object for output HDF5 L0B product
    band_path : str
        Frequency band path in L0B HDF5 product.
    tx_pol : str
        Tx polarization of the new echo product
    pulse_slice : slice
        Desired pulse slice for input HDF5 file.
    logger: logging.Logger
    keys: sequence of str
        Desired keys under TX group to be copied as long as
        they exist in input hdf5.
        Default is
        ('rangeLineIndex', 'radarTime', 'UTCtime', 'txPhase',
        'calType', 'chirpCorrelator')

    Notes
    -----
    'validSamplesSubSwath*' will be accounted for internally and
    thus shall be excluded from the `keys`.

    """
    # form list of keys excluding valid subswath
    keys_new = [k for k in keys if "validSamplesSubSwath" not in k]
    # get dataset names for valid subswath and appended them to the new list
    path = _join_paths(band_path, f'tx{tx_pol[0]}/')
    num_sbsw = fid_in[path + 'numberOfSubSwaths'][()]
    for n in range(1, num_sbsw + 1):
        keys_new.append(f'validSamplesSubSwath{n}')
    _copy_datasets_truncated(
        fid_in, fid_out, band_path, tx_pol, pulse_slice, logger, keys_new)


def cmd_line_parser():
    """ Command line parser """
    prs = argparse.ArgumentParser(
        'Generate a DBF/Science-like L0B product from DM2 L0B.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
    )
    prs.add_argument('--l0b', type=str, required=True, dest='l0b_file',
                     help='NISAR DM2 LOB product filename')
    prs.add_argument('--ant', type=str, required=True, dest='ant_file',
                     help='Antenna HDF5 filename.')
    grp_dem = prs.add_mutually_exclusive_group()
    grp_dem.add_argument('-d', '--dem', type=str, dest='dem_file',
                         default=None,
                         help='Filename for DEM raster ".tif" file.')
    grp_dem.add_argument('-r', '--ref-height', type=float, dest='ref_height',
                         default=0,
                         help=('Reference height (m) wrt WGS84 ellipsoid '
                               'if DEM raster is not provided.')
                         )
    prs.add_argument('--orbit', type=str, dest='orbit_file',
                     help='Filename of an orbit XML file. '
                     'Default is the one stored in L0B.')
    prs.add_argument('--attitude', type=str, dest='attitude_file',
                     help=('Filename of an attitude XML file. Default is '
                           'the one stored in L0B')
                     )
    prs.add_argument('-o', '--out-path', type=str, default='.',
                     dest='out_path',
                     help=('Output path for temporary files and '
                           'PNG plots if `--plot`')
                     )
    prs.add_argument('-p', '--product-name', type=str, dest='prod_name',
                     help=('Product science L0B HDF5 file and path name. '
                           'Default is input L0B filename with suffix '
                           '"_ONE_TAP_DBF_<utc-first>_<utc-last>" added prior '
                           'to the extension and is stored at the current '
                           'directory. First/last UTC is set by first/last '
                           'pulses to be processed.')
                     )
    prs.add_argument('--num-cpu', type=int, dest='num_cpu',
                     help=('Number of CPU/Workers used in range compression. '
                           'Default is all cores.')
                     )
    prs.add_argument('--no-rgcomp', action='store_true', dest='no_rgcomp',
                     help=('If set, the pulsewidth extension will not be '
                           'compensated at channel transition!')
                     )
    prs.add_argument('--num-rgl', type=int, dest='num_rgl',
                     default=8192,
                     help='Number of range lines in each AZ block')
    prs.add_argument('--comp-level-h5', type=int, dest='comp_level_h5',
                     default=4,
                     help='Compression level for HDF5 used for echo dataset.'
                     )
    prs.add_argument('--plot', action='store_true', dest='plot',
                     help='Plot one-tap DBFed echo ratser for each AZ block.')
    prs.add_argument('-m', '--multiplier', type=float, dest='multiplier',
                     help='DBFed echo multiplier prior to quantization.')
    prs.add_argument('-w', '--win-ped', type=float, dest='win_ped', default=1,
                     help=('Raised-cosine window pedestal used in '
                           'range comp. A value within [0, 1].')
                     )
    prs.add_argument('--calib', action='store_true', dest='calib',
                     help=('Apply complex calibration among RX channels. '
                           'The calibration factors/scalars if not provided, '
                           'will be obtained from inverse of relative '
                           'complex amplitude of caltones.'
                           )
                     )
    prs.add_argument('-a', '--amp-cal', nargs='*', type=float, dest='amp_cal',
                     help=('Calibration amplitude-only (linear) as '
                           'multipliers for all RX channels if "--calib". '
                           'Must be the same as number of RX channels. If '
                           'provided, will be used in place of inverse of '
                           'caltones!')
                     )
    prs.add_argument('--sample-delays', type=int, nargs='*',
                     help=('Relative integer sample delays of RX channels '
                           'wrt the first one in ascending RX order for '
                           'either the selected frequency band ("A" or "B") '
                           'or the very first of two ("A") if split spectrum. '
                           'The number of delays shall be equal to the total '
                           'number of RX channels minus one, excluding the '
                           'very first channel.')
                     )
    prs.add_argument('--sample-delays2', type=int, nargs='*',
                     help=('Relative integer sample delays of RX channels '
                           'wrt the first one in ascending RX order for '
                           'the second frequency band ("B") if split spectrum '
                           'and both bands are processed. The number of '
                           'delays shall be equal to the total number of RX '
                           'channels minus one, excluding the very first '
                           'channel.')
                     )
    prs.add_argument('--rx-antpat-calib', action='store_true',
                     help=('Calibrate one-way RX power pattern in elevation '
                           'to get rid of scalloping as a result of mosaicking'
                           )
                     )
    prs.add_argument('--max-p2p-ant', type=float,
                     help=('Max peak-to-peak dynamic range (dB) in RX antenna '
                           'pattern correction. Default is full antenna '
                           'pattern coverage of swath. A positive value used '
                           'only if `rx-antpat-calib`.')
                     )
    prs.add_argument('--time-start', type=float, default=0,
                     help=('Start time (seconds) wrt the first range line, '
                           '>= 0, to be processed. This is inclusive.')
                     )
    prs.add_argument('--time-stop', type=float,
                     help=('Stop time (seconds) wrt the first range line, '
                           '> 0, to be processed. Must be larger than '
                           '`time-start`. Its value will be limited by the '
                           'time of the last range line. This is exclusive. '
                           'The default is entire L0B.')
                     )
    return prs.parse_args()


def nisar_l0b_dm2_to_dbf(args):
    """Create NISAR one-tap DBFed L0B from DM2 L0B product"""
    # Const
    # list of rx/tx dataset keys/names to be resized/reproduced
    # in truncated output HDF5
    rx_keys = ('caltone', 'RD', 'WD', 'WL', 'basebandPhaseCorrection',
               'TRMDataWindow', 'attenuation')
    tx_keys = ('rangeLineIndex', 'radarTime', 'UTCtime', 'txPhase',
               'calType', 'chirpCorrelator')

    tic = time.time()
    if args.multiplier is not None and args.multiplier < 0:
        raise ValueError('Echo multiplier shall be a positive value!')

    # set logger
    logger = set_logger('DBF-L0B-FROM-DM2')

    plot = args.plot
    if plot and plt is None:
        logger.warning('No plot! Missing "matplotlib"!')
        plot = False

    # set number of CPU
    if args.num_cpu is None:
        n_cpu = os.cpu_count() or 1
    else:
        n_cpu = min(os.cpu_count() or 1, max(args.num_cpu, 1))
    logger.info(f'Number of CPUs used in FFT interpolation -> {n_cpu}')

    # parse antenna
    ant = AntennaParser(args.ant_file)
    # parse raw to get all products
    fid_in = h5py.File(args.l0b_file, 'r', swmr=True)
    raw = Raw(hdf5file=args.l0b_file)
    dm_flag = raw.identification.diagnosticModeFlag
    if dm_flag != 2:
        raise ValueError(
            'The input L0B is not a DM2 product! DM flag -> {dm_flag}!'
        )
    frq_pol = raw.polarizations
    # first frequency band and a co-pol
    freq_band = list(frq_pol.keys())[0]
    txrx_pol = frq_pol[freq_band][0]

    # check the RX calibration status
    if args.calib:
        amp_cal = args.amp_cal
        if amp_cal is not None:
            use_caltone = False
            # check number of RX channels match "amp_cal"
            n_rx_chanl = len(raw.getListOfRxTRMs(freq_band, txrx_pol))
            amp_cal = np.asarray(amp_cal)
            if amp_cal.size != n_rx_chanl:
                raise ValueError(
                    f'Number of values for "amp-cal" {amp_cal.size} is not'
                    f' the same as number of RX channels {n_rx_chanl}!'
                )
            logger.info(
                f'Apply user-provided amp calibration with values -> {amp_cal}'
            )
        else:
            use_caltone = True
            logger.info('Apply RX calibration based on inverse of slow-time'
                        ' averaged complex caltones!')
    else:  # No calibration, set cal amplitude to None!
        amp_cal = None

    # get ref epoch and build AZ slice generator
    epoch, azt_raw = raw.getPulseTimes(freq_band, txrx_pol[0])
    n_rgl_tot = azt_raw.size
    logger.info(f'Total available number of pulses in L0B -> {n_rgl_tot}')
    # Get start and end pulse index of input product to be processed
    # expect at least one range line to be processed!
    # Check the start time and get its pulse index
    azt_rel = azt_raw - azt_raw[0]
    if (args.time_start < 0) or not (args.time_start < azt_rel[-1]):
        raise ValueError(
            f'"time-start" shall be within [0, {azt_rel[-1]}) (sec)!')
    # start time is inclusive!
    start_pulse_idx = np.searchsorted(
        azt_rel, args.time_start, side='right') - 1
    # Check the stop time and get its pulse index
    if args.time_stop is None:
        stop_pulse_idx = n_rgl_tot
    else:
        if args.time_stop < azt_rel[start_pulse_idx + 1]:
            raise ValueError('"time-stop" shall be equal or larger than '
                             f'{azt_rel[start_pulse_idx + 1]} (sec)')
        # stop time is exclusive!
        stop_pulse_idx = np.searchsorted(
            azt_rel, args.time_stop, side='left') - 1
    logger.info('(start, stop) 0-based pulse index to be processed -> '
                f'({start_pulse_idx}, {stop_pulse_idx})')
    pulse_slice = slice(start_pulse_idx, stop_pulse_idx)
    # get number of pulses to be processed
    num_pulses = stop_pulse_idx - start_pulse_idx
    logger.info(f'Number of pulses for output L0B -> {num_pulses}')
    # Get start and end UTC datetime for output L0B products
    start_dt_utc = (epoch + TimeDelta(azt_raw[start_pulse_idx])).isoformat()
    end_dt_utc = (epoch + TimeDelta(azt_raw[stop_pulse_idx - 1])).isoformat()
    logger.info('(start, end) datetime for output L0B -> '
                f'({start_dt_utc}, {end_dt_utc})')

    # check the size of the sample delays and reverse the sign
    dset = raw.getRawDataset(freq_band, txrx_pol)
    size_delay = dset.shape[0] - 1
    sample_delays_all = [args.sample_delays, args.sample_delays2]
    for nn, (sample_delays, name_delay) in enumerate(
        zip(sample_delays_all, ["sample-delays", "sample-delays2"])
    ):
        if sample_delays is not None:
            if len(sample_delays) != size_delay:
                raise ValueError(
                    f'Size of "{name_delay}"={len(sample_delays)} must '
                    f'be {size_delay}!'
                )
            # reverse the sign for compensation of delays
            sample_delays_all[nn] = - np.asarray(sample_delays)
            logger.info(
                f'The amount of delay correction wrt RX # 1 for {name_delay} '
                f'-> {sample_delays_all[nn]}'
            )

    # parse orbit and attitude and check epoch
    if args.orbit_file is None:
        logger.info('Parsing orbit from L0B ...')
        orbit = raw.getOrbit()
    else:
        logger.info(f'Parsing orbit from "{args.orbit_file}"')
        orbit = load_orbit_from_xml(args.orbit_file)

    if orbit.reference_epoch != epoch:
        orbit.update_reference_epoch(epoch)

    if args.attitude_file is None:
        logger.info('Parsing attitude from L0B ...')
        attitude = raw.getAttitude()
    else:
        logger.info(f'Parsing attitude from "{args.attitude_file}"')
        attitude = load_attitude_from_xml(args.attitude_file)

    if attitude.reference_epoch != orbit.reference_epoch:
        attitude.update_reference_epoch(orbit.reference_epoch)

    # build dem
    if args.dem_file is not None:
        logger.info(f'Using DEM raster from file {args.dem_file}')
        dem = DEMInterpolator(Raster(args.dem_file))
    else:
        dem = DEMInterpolator(args.ref_height)
        logger.info(f'DEM Ref height is {dem.ref_height} (m)')

    if args.no_rgcomp:
        logger.warning('No rangecomp! Discontinuity at beam transition.')
    else:  # perform range conv and deconv
        logger.info('Perform range convolution and deconvolution!')
        logger.info('Raised-cosine window pedestal used in range comp'
                    f' -> {args.win_ped}')

    # form BFPQ LUT for uniform quantizer to be used for output product
    nbits = 16
    bfpq_uq = build_uniform_quantizer_lut_l0b(nbits)
    max_valid_int = 2**(nbits - 1) - 1

    def utc2filename(dt_utc: str) -> str:
        """Datetime UTC string into filename string"""
        return datetime.fromisoformat(dt_utc[:19]).strftime('%Y%m%dT%H%M%S')

    # get in/out files and file objects
    p_in = Path(args.l0b_file)
    out_path = Path(args.out_path)
    if args.prod_name is None:
        utc_first = utc2filename(start_dt_utc)
        utc_last = utc2filename(end_dt_utc)
        suffix = f'_ONE_TAP_DBF_{utc_first}_{utc_last}.h5'
        file_out = p_in.stem + suffix
    else:
        file_out = args.prod_name
    logger.info(f'Filename of output 1-tap DBF product -> "{file_out}"')
    fid_out = h5py.File(file_out, 'w')

    # copy the entire data for metadata, low_res telemetry
    grp_rrsd = fid_out.require_group(raw.ProductPath)
    fid_in.copy(raw.TelemetryPath, grp_rrsd)
    fid_in.copy(raw.MetadataPath, grp_rrsd)
    # copy entire high-rate telemetry group if exists
    hrt_path = raw.TelemetryPath.replace('low', 'high')
    try:
        fid_in.copy(hrt_path, grp_rrsd)
    except KeyError as err:
        logger.warning(f'Missing group "{hrt_path}" in input L0B!'
                       f' Detailed Error -> "{err}"')
    # copy the entire identification into the output product
    # but modify diagnostic mode flags from DM2 to DBF
    # as well as update start and end zero-Doppler datetime!
    grp_root = fid_out.require_group(raw.RootPath)
    fid_in.copy(raw.IdentificationPath, grp_root)
    grp_ident = fid_out[raw.IdentificationPath]
    grp_ident['diagnosticModeFlag'][()] = np.uint8(0)
    grp_ident['zeroDopplerStartTime'][()] = np.bytes_(start_dt_utc)
    grp_ident['zeroDopplerEndTime'][()] = np.bytes_(end_dt_utc)
    # copy entire swath except for TxRx echo products
    copy_swath_except_echo_h5(fid_in, fid_out, raw.SwathPath, frq_pol,
                              tx_keys=tx_keys, rx_keys=rx_keys)

    # get DBF sampling rate
    fs_dbf = raw.getSampleRateDBF(freq_band, txrx_pol)
    logger.info(f'DBF sampling rate for RD/WD/WL (MHz) -> {fs_dbf * 1e-6:.3f}')

    # loop over all products and bands
    # form a vector of range line slices used for all products
    rgl_slices = list(
        slice_gen(num_pulses, args.num_rgl, idx_start=start_pulse_idx)
    )
    logger.info(f'Number of AZ blocks -> {len(rgl_slices)}')
    # Build slices for output L0B that always starts from zero index
    if start_pulse_idx == 0:
        rgl_slices_out = rgl_slices
    else:
        rgl_slices_out = list(
            slice_gen(num_pulses, args.num_rgl, idx_start=0)
        )

    for freq_band, sample_delays in zip(frq_pol, sample_delays_all):
        # group path for frequency band
        band_path = raw.BandPath(freq_band) + '/'

        for txrx_pol in frq_pol[freq_band]:
            logger.info(
                f'Processing frequency band {freq_band} and Pol '
                f'{txrx_pol} ...')
            # get slow-time invariant calib coefs if calib requested
            if args.calib:
                if use_caltone:
                    # use mean caltone over all range lines rather than per
                    # AZ blocks to avoid introducing any undesired slow-time
                    # variation affecting AZ impulse response.
                    # Generally speaking, caltone should stay stable at least
                    # within a minute data acquisition!
                    caltones = raw.getCaltone(freq_band, txrx_pol)
                    cal_avg = caltones.mean(axis=0)
                    logger.info(f'Averaged Caltones -> {cal_avg}')
                    # check amplitude to be nonzero
                    cal_avg_amp = abs(cal_avg)
                    if not (cal_avg_amp.min() > 0):
                        raise RuntimeError('Zero Caltone values encountered!')
                    # peak normalized and inverse the complex caltones coeffs
                    # whose magnitude is a value within (0, 1]
                    amp_cal = cal_avg_amp.min() / cal_avg
                logger.info(f'Final calibration multipliers -> {amp_cal}')

            # copy truncated tx/rx datasets per desired rx/tx keys
            copy_tx_datasets_truncated(fid_in, fid_out, band_path, txrx_pol[0],
                                       pulse_slice, logger, keys=tx_keys)
            copy_rx_datasets_truncated(fid_in, fid_out, band_path, txrx_pol,
                                       pulse_slice, logger, keys=rx_keys)
            # product group path
            prod_grp = band_path + f'tx{txrx_pol[0]}/rx{txrx_pol[1]}/'

            # get transition points between beams in (EL, AZ) for only
            # active channels
            list_rx_active = raw.getListOfRxTRMs(freq_band, txrx_pol)
            logger.info(f'List of active RX channels -> {list_rx_active}')

            el_trans, az_trans = ant.locate_beams_overlap(txrx_pol[1])
            el_trans = el_trans[list_rx_active[:-1] - 1]
            logger.info(
                f'EL angles @ beams transitions -> {np.rad2deg(el_trans)}'
                ' (deg)')
            logger.info('AZ angle for all beams transitions -> '
                        f'{np.rad2deg(az_trans)} (deg)')

            # parse echo and slant ranges
            sr = raw.getRanges(freq_band, txrx_pol[0])
            dset = raw.getRawDataset(freq_band, txrx_pol)

            # check number of channels with antenna beam numbers
            num_beams = ant.num_beams(txrx_pol[1])
            num_chanl = dset.shape[0]
            if num_beams != num_chanl:
                raise RuntimeError(
                    f'Mismatch between number of antenna beams {num_beams} and'
                    f' L0B RX channels {num_chanl} for Pol {txrx_pol[1]}!'
                )
            # create a placeholder for echo product and use uniform-quantizer
            # BFPQ LUT in place of BFPQ ones.
            dset_prod_out = create_echo_dataset_h5(
                fid_in, fid_out, band_path, txrx_pol, (num_pulses, sr.size),
                dset.dtype_storage, args.comp_level_h5)

            bfpq_path = prod_grp + 'BFPQLUT'
            fid_out[bfpq_path][:] = bfpq_uq
            # Get valid subswath to fill in TX gap regions with zero
            # due to non-mitigated strong TX Cal loop-back chirps.
            sbsw = raw.getSubSwaths(freq_band, txrx_pol[0])

            # if RX Antenna pattern calibration is required then parse
            # EL-cut patterns and get peak locations in EL direction
            # for all beams to be used along with `el_trans` in 2nd-order
            # polyfitting of EL power pattern (dB) as a function of
            # angle/slant range.
            if args.rx_antpat_calib:
                logger.info(
                    f'Calibrate for EL RX power pattern of Pol "{txrx_pol[1]}"'
                )
                el_peaks, az_peak = ant.locate_beams_peak(txrx_pol[1])
                logger.info(
                    f'EL angles @ beams transitions -> {np.rad2deg(el_peaks)}'
                    ' (deg)')
                logger.info('AZ angle for all beams transitions -> '
                            f'{np.rad2deg(az_peak)} (deg)')
                el_cuts = ant.el_cut_all(txrx_pol[1])
                # get el angle at lower and upper edge of 2-way HPBW
                # of the first and last beam respectively to cover three
                # points required per beam including
                el_first, el_last = _el_swath_edges(el_cuts)
                logger.info(
                    'EL (first ,last) in pattern polyfitting (deg, deg) -> '
                    f'({np.rad2deg(el_first):.3f}, {np.rad2deg(el_last):.3f})'
                )
                if not (el_first < el_peaks[0]):
                    raise ValueError(
                        f'First EL angle {np.rad2deg(el_first):.3f} (deg) is '
                        'not smaller than the peak of the first beam '
                        f'{np.rad2deg(el_peaks[0]):.3f} (deg)! '
                        'Not enough EL antenna pattern coverage!'
                    )
                if not (el_last > el_peaks[-1]):
                    raise ValueError(
                        f'Last EL angle {np.rad2deg(el_last):.3f} (deg) is '
                        'not larger than the peak of the last beam '
                        f'{np.rad2deg(el_peaks[-1]):.3f} (deg)! '
                        'Not enough EL antenna pattern coverage!'
                    )
                # form list of EL angles to be converted into slant ranges
                # [first, peak1, intersect1-2, peak2, intersect2-3, ..., last ]
                # 2N + 1 values for N beams
                el_points = np.zeros(2 * num_beams + 1)
                el_points[0] = el_first
                el_points[-1] = el_last
                el_points[1:-1:2] = el_peaks
                el_points[2:-2:2] = el_trans
                logger.info('EL points used in EL pattern polyfitting (deg) '
                            f'-> {np.rad2deg(el_points)}')
                # report threshold
                if args.max_p2p_ant is not None:
                    logger.warning(
                        'Dynamic range of RX antenna pattern correction will '
                        f'be limited to {args.max_p2p_ant} (dB).'
                    )

            # get chirp sampling rate for computing updated DBF WD/WL
            # XXX For NISAR modes with zero bandwidth (no TX), the chirp
            # slope is assumed to be `-fs / (1.2 * pw)`!
            fs, slope, pw = _get_chirp_parameters(raw, freq_band, txrx_pol[0])
            nrgb_pw = np.ceil(fs * pw).astype(int)
            # form chirp reference if seamless/rangecomp requested
            if not args.no_rgcomp:
                chirp_ref = np.asarray(form_linear_chirp(slope, pw, fs))
            # define a function to rescale range bin limits/coverage per
            # channel based on DBF sampling rate for updating WD/WL in
            # single-tap DBF.

            def _rescale2dbf(rgb_limits):
                return np.ceil(
                    (fs_dbf / fs) * np.asarray(rgb_limits)).astype(int)
            # parse WD/RD values for all channels over all lines
            wd = raw.getWD(freq_band, txrx_pol)
            rd = raw.getRD(freq_band, txrx_pol)
            # get the first DWP over all range lines
            # Note that first channel always carries min DWP (opens first!).
            dwp_first = rd[:, 0] + wd[:, 0]

            # create a temp file for memmap of multi-channel complex
            # decoded echo to avoid possible memory allocation issue
            fid_tmp = tempfile.NamedTemporaryFile(
                suffix=f'_dm2_freq{freq_band}_pol{txrx_pol}.c8',
                dir=out_path)
            # form a numpy 3-D memmap AZ block shared by all AZ blocks
            num_rgl = rgl_slices[0].stop - rgl_slices[0].start
            dset_azblk = np.memmap(
                fid_tmp, mode='w+',
                shape=(num_chanl, num_rgl, sr.size),
                dtype=dset.dtype)
            # loop over AZ blocks
            for n_blk, (rgl_slice, rgl_slice_o) in enumerate(
                    zip(rgl_slices, rgl_slices_out), start=1):
                logger.info(f'Processing AZ block # {n_blk} ...')
                logger.info(f'Processing input rangelines -> {rgl_slice}')
                num_rgl = rgl_slice.stop - rgl_slice.start
                # fill in 3-D memmap complex array with decoded echo,
                # one channel at a time to avoid memory issue!
                for cc in range(num_chanl):
                    dset_azblk[cc, :num_rgl] = dset[cc, rgl_slice, :]
                # fill in TX gap regions with zeros in place
                fill_gaps(dset_azblk[:, :num_rgl], sbsw[:, rgl_slice])

                # Adjust the relative integer sample delays in place
                # over all channels within an AZ block
                _adjust_delays_in_place(dset_azblk[:, :num_rgl], sample_delays)

                # mid AZ time at the center of the AZ block
                azt_mid = azt_raw[rgl_slice].mean()

                # Compute RX ANT Power Pattern calibration if requested
                if args.rx_antpat_calib:
                    if plot:
                        plot_name = out_path.joinpath(
                            f'Plot_OneTap_DBF_EL_RxAntPat_Freq{freq_band}_'
                            f'Pol{txrx_pol}_AzBlock{n_blk}.png'
                        )
                    else:
                        plot_name = None
                    inv_rxpat_1w = _reconstruct_inverse_el_magpat_full_swath(
                        el_cuts, el_points, az_trans, azt_mid, sr, orbit,
                        attitude, dem, max_p2p_ant=args.max_p2p_ant,
                        plot_name=plot_name)
                else:  # No RX ANT Pattern Correction
                    inv_rxpat_1w = None

                if args.no_rgcomp:  # simply perform mosaicking
                    echo_dbf, rgb_limits = dbf_onetap_from_dm2(
                        dset_azblk[:, :num_rgl], azt_mid, el_trans, az_trans,
                        sr, orbit, attitude, dem, cal_coefs=amp_cal
                    )
                    # Apply the inverse of RX EL amplitude pattern to the echo
                    # if requested
                    if args.rx_antpat_calib:
                        # XXX convolve square of inverse of `inv_rxpat_1w` by
                        # pulsewidth prior to echo multiplication to include
                        # pulse extension!
                        # Alternatively, apply ANT correction after range comp
                        # as part of RSLC workflow later on (preferred)!
                        inv_rxpat_1w[...] = 1 / np.sqrt(
                            fftconvolve(
                                inv_rxpat_1w**(-2),
                                (1 / nrgb_pw) * np.ones(nrgb_pw),
                                mode='same'
                            )
                        )
                        echo_dbf *= inv_rxpat_1w

                else:  # perform range conv and deconv while mosaicking
                    logger.info('Perform range convolution and deconvolution!')
                    echo_dbf, rgb_limits = dbf_onetap_from_dm2_seamless(
                        dset_azblk[:, :num_rgl], chirp_ref, azt_mid, el_trans,
                        az_trans, sr, orbit, attitude, dem, n_cpu,
                        ped_win=args.win_ped, cal_coefs=amp_cal,
                        inv_antpat_el=inv_rxpat_1w)
                    # scale echo by sqrt(BW * PW) to remove compression gain
                    # and to preserve input dynamic range
                    scalar_cg = np.sqrt(abs(slope) * pw ** 2)
                    logger.info('Remove compression amp gain (linear)'
                                f' -> {scalar_cg}.')
                    echo_dbf /= scalar_cg

                # plot float-point DBFed raster per AZ block
                if plot:
                    plt.figure(figsize=(8, 6))
                    plt.imshow(abs(echo_dbf), cmap='grey', aspect='auto')
                    plt.xlabel('Range Bins (-)')
                    plt.ylabel('Azimuth Bins (-)')
                    plt.title(f'One-Tap DBFed Echo for AZ Block # {n_blk}')
                    plot_name = out_path.joinpath(
                        f'Plot_OneTap_DBF_Echo_Freq{freq_band}_'
                        f'Pol{txrx_pol}_AzBlock{n_blk}.png'
                    )
                    logger.info('Filename of DBFed Raster Plot -> '
                                f'{plot_name}')
                    plt.savefig(plot_name)
                    plt.close()

                # scaled echo is requested
                if args.multiplier is not None:
                    logger.info(f'DBFed Echo for AZ block {n_blk} is '
                                f'multiplied by -> {args.multiplier}')
                    echo_dbf *= args.multiplier
                # quantize echo
                echo_dbf = echo_dbf.view('f4').round()

                # check the max amplitude for possible overflow
                max_echo_dbf = abs(echo_dbf).max()
                if max_echo_dbf > max_valid_int:
                    logger.warning(
                        f'Overflow! Max amp of DBFed echo -> {max_echo_dbf}!')
                    scalar_echo = max_valid_int / max_echo_dbf
                    logger.warning(
                        f'Suggested DBF echo multiplier -> {scalar_echo}')

                # store echo data per AZ block
                dset_prod_out.write_direct(echo_dbf.astype(
                    'uint16').view(dset.dtype_storage), dest_sel=rgl_slice_o)

                # compute WD/WL values to be updated per single-tap DBF
                # use mid range line values
                rgb_limits_dbf = _rescale2dbf(rgb_limits)
                rgl_mid = (rgl_slice.start + rgl_slice.stop) // 2
                rd_new = rd[rgl_mid]
                wd_new, wl_new = _wd_wl_single_tap_dbf(
                    rgb_limits_dbf, dwp_first[rgl_mid], rd_new)
                logger.info(f'New WDs per AZ block # {n_blk} -> {wd_new}')
                logger.info(f'New WLs per AZ block # {n_blk} -> {wl_new}')
                # update HDF5 datasets RD/WD/WL per AZ block
                grp_prod = dset_prod_out.parent
                grp_prod['WD'][rgl_slice_o] = wd_new
                grp_prod['WL'][rgl_slice_o] = wl_new
                grp_prod['RD'][rgl_slice_o] = rd_new

            # destroy memmap and close the temp file
            del dset_azblk, fid_tmp

    # close in/out HDF5 files
    fid_in.close()
    fid_out.close()

    logger.info(f'Elapsed time (sec) -> {time.time() - tic:.1f}')


def _get_chirp_parameters(raw, freq_band, txrx_pol, slope_sign=-1):
    """
    Get baseband chirp parameters from Raw for a desired
    frequency band and polarization.

    This function extracts chirp parameters from the input L0B product when
    possible. However, in the case of a no-transmit (noise-only) channel, the
    chirp slope parameter in the product metadata will be zero. In this case,
    the function estimates the chirp slope of the nominal Tx waveform by
    assuming an oversampling ratio of 1.2 for NISAR products.

    Parameters
    ----------
    raw : nisar.products.readers.Raw
        Raw parser of L0B product.
    freq_band : str
        Frequency band character "A" or "B".
    txrx_pol : str
        Transmit-receive polarization such as "HH", "HV", etc.
    slope_sign : {-1, 1}
        Sign of the chirp slope. Default is down chirp (-1).
        This is simply used if the chirp slope or bandwidth is
        set to zero for some NISAR modes.
        In this case, the chirp slope is assumed to be
        `slope_sign * fs / (1.2 * pw)` where `fs` and `pw` are
        the chirp sampling rate and pulsewidth, respectively.

    Returns
    -------
    fs : float
        Chirp sampling rate in Hz
    slope : float
        Chirp slope in Hz/seconds
    pw : float
        Chirp pulsewidth in seconds

    """
    _, fs, slope, pw = raw.getChirpParameters(freq_band, txrx_pol[0])
    if np.isclose(slope, 0.0):
        slope = slope_sign * fs / (1.2 * pw)
    return fs, slope, pw


def _adjust_delays_in_place(dset, sample_delays):
    """
    Adjust relative sample delays over all RX channels except
    the very first channel in place.

    Parameters
    ----------
    dset : array of complex float
        3-D array of raw decoded data with shape
        (RX channels, range lines, range bins) to be modified in place.
    sample_delays : array of int or None
        Array-like signed integers representing relative sample delay of
        RX channels wrt the very first one to be compensated in range
        direction. The order of channels is ascending.
        The size of array shall be `RX channels - 1`.
        If None, no delay adjustment will be applied.

    """
    if sample_delays is not None:
        for cc, delay in enumerate(sample_delays, start=1):
            if delay != 0:
                dset[cc] = np.roll(dset[cc], shift=delay, axis=-1)


def _el_swath_edges(el_cuts):
    """
    Get approximate one-way HPBW lower/upper edge for the first/last
    beam within the swath if there is enough angular coverage in EL.

    Parameters
    ----------
    el_cuts : AntPatCut

    Returns
    -------
    float
        EL angle (radians) of lower edge of the first beam
    float
        EL angle (radians) of upper edge of the last beam

    """
    # one-way HPBW threshold
    threshold = 10 ** (-3 / 20)
    # first beam (lower edge)
    mag_first = abs(el_cuts.copol_pattern[0])
    idx_peak = np.nanargmax(mag_first)
    mag_hpbw = mag_first[idx_peak] * threshold
    idx_first = np.nanargmin(abs(mag_first[:idx_peak] - mag_hpbw))
    # last beam (upper edge)
    mag_last = abs(el_cuts.copol_pattern[-1])
    idx_peak = np.nanargmax(mag_last)
    mag_hpbw = mag_last[idx_peak] * threshold
    idx_last = np.nanargmin(abs(mag_last[idx_peak:] - mag_hpbw)) + idx_peak

    return el_cuts.angle[idx_first], el_cuts.angle[idx_last]


def _amp2db(amp: np.ndarray) -> np.ndarray:
    return 20 * np.log10(abs(amp))


def _reconstruct_inverse_el_magpat_full_swath(
        el_cuts, el_points, az_bs, az_time, sr, orbit,
        attitude, dem, max_p2p_ant=None, plot_name=None):
    """
    Helper function to build mosaicked (1-tap DBFed) relative (peak-normazlied)
    inverse of one-way EL magnitude pattern as a function of slant range to be
    used for removing one-way RX antenna pattern from mosaciked DM2.
    This will mitigate scalloping effect in the range profiles of echo.

    Parameters
    ----------
    el_cuts : nisar.products.readers.antenna.AntPatCut
        It contains EL-cut patterns for all beams.
    el_points : np.ndarray(float)
        EL angles covering start, peaks, transitions, and end of swath
        all in radians. The size is `2N + 1` where `N` is number of beams.
        Bascialy, three distinct points per beam including the intersections
        common among adjacent beams.
    az_bs : float
        Common azimuth boresight angle over all beams in radians
    az_time : float
        AZ time w.r.t. to orbit epoch in seconds
    sr : isce3.core.Linspace
        Slant ranges over entire swath.
    orbit : isce3.core.Orbit
    attitude : isce3.core.Attitude
    dem : isce3.geometry.DEMInterpolator
    max_p2p_ant : float, optional
        Max peak-to-peak dynamic range of RX antenna pattern in (dB).
        Default is full dynamic range over entire swath.
    plot_name : str, optional
        If provided, it will generate PNG plot of peak-normalized
        one-tap DBFed Mosaicked one-way EL Antenna Power Pattern.

    Returns
    -------
    np.ndarray(float)
        1-D array of inverse of relative EL magnitude pattern (linear)
        with the same size as `sr`.

    """
    if max_p2p_ant is not None and not (max_p2p_ant > 0):
        raise ValueError('"max_p2p_ant" must be a positive value!')
    # Get slant ranges at beams transition
    pos, vel = orbit.interpolate(az_time)
    quat = attitude.interpolate(az_time)
    # Pass a dummy wavelength=1 since it doesn't affect the result.
    sr_points, _, _ = ant2rgdop(el_points, az_bs, pos, vel, quat, 1, dem)
    # convert slant ranges to range bins for beam limits
    rgb_points = np.round((sr_points - sr.first) / sr.spacing).astype(int)
    # replace first and the last by 0 and sr.size
    rgb_points[0] = 0
    rgb_points[-1] = sr.size
    # convert slant range from meters into km for polyfitting
    sr_points_km = 1e-3 * sr_points
    magpat1w = np.zeros(sr.size, dtype='f8')
    # loop over beams
    for cc, i_start in enumerate(range(0, el_points.size - 1, 2)):
        # get three points per beam
        i_slice = slice(i_start, i_start + 3)
        # interpolate EL pattern for three points of EL within a beam
        # and get its power in dB.
        el_gain_point3 = _amp2db(
            np.interp(el_points[i_slice],
                      el_cuts.angle,
                      el_cuts.copol_pattern[cc])
        )
        # perform 2-order polyfit of power (dB) as a function sr (km)
        pf_coef_db_km = np.polyfit(
            sr_points_km[i_slice], el_gain_point3, deg=2)
        # now get magnitude pattern over all slant range covering a beam
        # by evaluating 2-order polyfit of power (dB) as a functon sr (km)
        rgb_slice = slice(rgb_points[i_start], rgb_points[i_start + 2])
        el_pow_db = np.polyval(pf_coef_db_km, 1e-3 * np.asarray(sr[rgb_slice]))
        # store the magnitude in dB per beam
        magpat1w[rgb_slice] = el_pow_db
    # limit dynamic range of antenna pattern correction,
    # that is min value wrt the peak value.
    pk_magpat1w = np.nanmax(magpat1w)
    if max_p2p_ant is not None:
        threshold = pk_magpat1w - max_p2p_ant
        magpat1w[magpat1w < threshold] = threshold
    # plot power pattern prior to inversion
    if plot_name is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(np.asarray(sr) * 1e-3, magpat1w - pk_magpat1w)
        plt.xlabel('Slant Range (km)')
        plt.ylabel('Relative EL Power Pattern (dB)')
        plt.title(
            'One-tap DBFed/Mosaicked Peak-Normalized RX EL Power Pattern')
        plt.grid(True)
        plt.savefig(plot_name)
        plt.close()
    # convert to linear scale check the min value to be non-zero
    magpat1w[:] = 10 ** (magpat1w / 20)
    min_mag = magpat1w.min()
    if np.isclose(min_mag, 0):
        raise ValueError(
            'The one-way RX EL power pattern contain zero value(s)! '
            'Consider setting "max-p2p-ant" option to limit dynamic range.'
        )
    # inverse and peak normalized the magnitude to be multiplied with
    # complex echo
    magpat1w[:] = min_mag / magpat1w
    return magpat1w


def _wd_wl_single_tap_dbf(rgb_limits_dbf, dwp_first, rd_all):
    """
    Build DBFed WD/WL values per range bin limits for a single-tap DBF.
    Note that range limits are @ DBF clock rate.

    Parameters
    ----------
    rgb_limits_dbf : np.ndarray(int)
        1-D array of range bin limits @DBF clock rate for all beams (channels)
        in ascending order with size `channels + 1`.
        For instance, the respective [start, stop] range bins for channel
        `i` are indices `[i - 1, i]`.
    dwp_first : uint32
        First channel data window position (DWP) @ DBF clock rate.
    rd_all : np.ndarray(uint32)
        RD values for all channels @ DBF clock rate.
        The size is equal to number of RX channels.

    Returns
    -------
    wd : np.array(uint32)
        Single-tap DBF WD (start of window) values for all channels.
        The size equals to the number of RX channels.
    wl : np.array(uint32)
        Single-tap DBF WL (length of window) values for all channels.
        The size equals to the number of RX channels.

    """
    n_channel = rd_all.size
    wd = np.zeros(n_channel, dtype='uint32')
    wl = np.zeros_like(wd)
    for nn in range(n_channel):
        wd[nn] = rgb_limits_dbf[nn] + dwp_first - rd_all[nn]
        wl[nn] = rgb_limits_dbf[nn + 1] - rgb_limits_dbf[nn]
    return wd, wl


if __name__ == '__main__':
    nisar_l0b_dm2_to_dbf(cmd_line_parser())
