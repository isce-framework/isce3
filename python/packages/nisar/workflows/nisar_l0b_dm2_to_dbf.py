#!/usr/bin/env python3
"""
Produce NISAR DBF/Science-like L0B product from NISAR DM2 L0B
"""
import argparse
from pathlib import Path
import os
import time

import numpy as np
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
from isce3.focus import fill_gaps
from nisar.workflows.helpers import build_uniform_quantizer_lut_l0b, slice_gen


def copy_swath_except_echo_h5(fid_in, fid_out, swath_path, frq_pol):
    """
    Copy all groups and datasets under swath from input HDF5 to output
    HDF5 excpet for echo products.

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

    """
    for freq_band in frq_pol:
        # build band path
        band_path = os.path.join(swath_path, f'frequency{freq_band}/')
        # create freq band group for output product
        grp_band = fid_out.require_group(band_path)
        # list of all TxRx products
        txrx_pols = frq_pol[freq_band]
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
                            if rx_item not in txrx_pols:
                                # copy all RX fields except echo product
                                fid_in.copy(rx_path + rx_item, grp_rx)

                    else:  # copy the rest of TX as it is
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
    rx_path = os.path.join(band_path, f'tx{txrx_pol[0]}/rx{txrx_pol[1]}/')
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
                     help=('Output path for PNG plots if `--plot`')
                     )
    prs.add_argument('-p', '--product-name', type=str, dest='prod_name',
                     help=('Product science L0B HDF5 file and path name. '
                           'Default is input L0B filename with suffix '
                           '"OneTap_DBF" added prior to the extension and is '
                           'stored at the current directory.')
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
                     default=5000,
                     help='Number of range lines in each AZ block')
    prs.add_argument('--comp-level-h5', type=int, dest='comp_level_h5',
                     default=4,
                     help='Compression level for HDF5 used for echo dataset.'
                     )
    prs.add_argument('--plot', action='store_true', dest='plot',
                     help='Plot one-tap DBFed echo ratser for each AZ block.')
    prs.add_argument('-m', '--multiplier', type=float, dest='multiplier',
                     help='DBFed echo multipler prior to quantization.')
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
    return prs.parse_args()


def nisar_l0b_dm2_to_dbf(args):
    """Create NISAR one-tap DBFed L0B from DM2 L0B product"""

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
    amp_cal = args.amp_cal
    if args.calib:
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
    # get ref epoch and build AZ slice generator
    epoch, azt_raw = raw.getPulseTimes(freq_band, txrx_pol[0])
    n_rgl_tot = azt_raw.size

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

    # get in/out files and file objects
    p_in = Path(args.l0b_file)
    out_path = Path(args.out_path)
    if args.prod_name is None:
        suffix = '_OneTap_DBF.h5'
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
    grp_root = fid_out.require_group(raw.RootPath)
    fid_in.copy(raw.IdentificationPath, grp_root)
    grp_ident = fid_out[raw.IdentificationPath]
    grp_ident['diagnosticModeFlag'][()] = np.uint8(0)
    # copy entire swath except for TxRx echo products
    copy_swath_except_echo_h5(fid_in, fid_out, raw.SwathPath, frq_pol)

    # loop over all products and bands
    # form a vector of range line slices used for all products
    rgl_slices = list(slice_gen(n_rgl_tot, args.num_rgl))
    logger.info(f'Number of AZ blocks -> {len(rgl_slices)}')

    for freq_band in frq_pol:
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
                    # within a minute data aquisition!
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
                fid_in, fid_out, band_path, txrx_pol, dset.shape[1:],
                dset.dtype_storage, args.comp_level_h5)

            bfpq_path = prod_grp + 'BFPQLUT'
            fid_out[bfpq_path][:] = bfpq_uq

            # Get valid subswath to fill in TX gap regions with zero
            # due to non-mitigated strong TX Cal loop-back chirps.
            sbsw = raw.getSubSwaths(freq_band, txrx_pol[0])

            # loop over AZ blocks
            for n_blk, rgl_slice in enumerate(rgl_slices, start=1):
                logger.info(f'Processing AZ block # {n_blk} ...')

                # fill in TX gap regions with zeros in place
                dset_azblk = dset[:, rgl_slice, :]
                fill_gaps(dset_azblk, sbsw[:, rgl_slice])

                # mid AZ time at the center of the AZ block
                azt_mid = azt_raw[rgl_slice].mean()

                if args.no_rgcomp:  # simply peform mosaicking
                    echo_dbf = dbf_onetap_from_dm2(
                        dset_azblk, azt_mid, el_trans, az_trans,
                        sr, orbit, attitude, dem, cal_coefs=amp_cal
                    )
                else:  # perform range conv and deconv while mosaicking
                    logger.info('Perform range convolution and deconvolution!')
                    chirp_ref = raw.getChirp(freq_band, txrx_pol[0])
                    echo_dbf = dbf_onetap_from_dm2_seamless(
                        dset_azblk, chirp_ref, azt_mid, el_trans,
                        az_trans, sr, orbit, attitude, dem, n_cpu,
                        ped_win=args.win_ped, cal_coefs=amp_cal)
                    # scale echo by sqrt(BW * PW) to remove compression gain
                    # and to preserve input dynamic range.
                    _, _, rate, pw = raw.getChirpParameters(
                        freq_band, txrx_pol[0])
                    scalar_cg = np.sqrt(abs(rate) * pw ** 2)
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
                    'uint16').view(dset.dtype_storage), dest_sel=rgl_slice)

    # close in/out HDF5 files
    fid_in.close()
    fid_out.close()

    logger.info(f'Elapsed time (sec) -> {time.time() - tic:.1f}')


if __name__ == '__main__':
    nisar_l0b_dm2_to_dbf(cmd_line_parser())
