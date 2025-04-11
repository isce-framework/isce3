#!/usr/bin/env python3
"""
Produce NISAR science L0B from NISAR DM1 L0B product.
"""
import argparse
from pathlib import Path
import os
import time
from warnings import warn

import numpy as np
from scipy import fft
import h5py
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from nisar.products.readers.Raw import Raw
from nisar.log import set_logger
from isce3.core import Linspace, speed_of_light
from isce3.signal import build_multi_rate_fir_filter
from nisar.workflows.helpers import build_uniform_quantizer_lut_l0b, slice_gen


def cmd_line_parser():
    """ Command line parser """
    prs = argparse.ArgumentParser(
        'Generate science-like L0B product from DM1 L0B.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@'
    )
    prs.add_argument('l0b_file', type=str,
                     help='L0B DM1 filename')

    prs.add_argument('-o', '--out-path', type=str, default='.',
                     dest='out_path',
                     help='Output path for PNG plots if `--plot`.'
                     )
    prs.add_argument('-p', '--product-name', type=str, dest='prod_name',
                     help=('Product science L0B HDF5 file and path name. '
                           'Default is input L0B filename with suffix '
                           '"Science" added prior to the extension and is '
                           'stored at the current directory.')
                     )
    prs.add_argument('--num-cpu', type=int, dest='num_cpu',
                     help=('Number of CPU/Workers used in filtering.'
                           ' Default is all cores.')
                     )
    prs.add_argument('--comp-level-h5', type=int, dest='comp_level_h5',
                     default=4, help='Compression level for HDF5 used for'
                     ' echo dataset.')
    prs.add_argument('--num-rgl', type=int, dest='num_rgl',
                     default=4000,
                     help='Number of range lines in each AZ block')
    prs.add_argument('--plot', action='store_true', dest='plot',
                     help='Plot science echo ratser for each AZ block.')

    prs.add_argument('--ovsf-rg', type=float, dest='ovsf_rg',
                     help=('Over sampling factor (ovsf) in range. '
                           'A value > 1. The output sampling rate for each '
                           'frequency band is determined by product of '
                           'chirp bandwidth and ovsf. Default is set based'
                           ' on the NISAR TX chirp bandwidth.')
                     )
    prs.add_argument('--sign-mag', action='store_true', dest='sign_mag',
                     help=('Assumed signed-magnitude representation of '
                           'binary echo data. Default is twos complement! '
                           'This will affect order of negative values in '
                           'BFPQLUT decoder.')
                     )
    prs.add_argument('--nbits', type=int, dest='nbits', default=12,
                     help='Number of bits representing full dynamic'
                     ' range of valid encoded echo samples.')
    return prs.parse_args()


def range_filtering_per_az_block(echo, coefs_fft, flt, n_cpu):
    """
    A helper function for range filtering of 2-D echo (azimuth by range)
    in range direction using FFT of filter coeffs along with filter
    parameters.

    Parameters
    ----------
    echo : np.ndarray(complex)
        2-D array of complex echo with shape (azimuth by range)
    coefs_fft : np.ndarray(complex)
        FFT of the filter coefficients. The number of FFT bins is assumed
        to be equal or larger full convolution length after filtering
        that is `echo.shape[0] * flt.upfact + flt.numtaps - 1`.
    flt : isce3.signal.MultiRateFirFilter
        Multi-rate filter parameters of the `coefs_fft`.
    n_cpu : int
        Number of CPUs/Workers used in filtering process

    Returns
    -------
    np.ndarray(complex)
        2-D filtered array with shape (azimuth by range).
        The azimuth dimension has the same size as echo!

    """
    n_azb, n_rgb = echo.shape
    nfft = coefs_fft.size
    # check the nfft is at least min required one based on
    # a full convolution during filtering.
    nfft_min = n_rgb * flt.upfact + flt.numtaps - 1
    if nfft < nfft_min:
        raise ValueError(
            f'FFT bins in coefs is smaller than min {nfft_min}')
    if flt.upfact > 1:  # upsample input
        n_rgb_up = n_rgb * flt.upfact
        echo_up = np.zeros((n_azb, n_rgb_up), dtype=echo.dtype)
        echo_up[:, ::flt.upfact] = echo
        slice_rg = slice(
            flt.groupdelsamp, flt.groupdelsamp + n_rgb_up, flt.downfact)
    else:  # no upsampling
        echo_up = echo
        slice_rg = slice(
            flt.groupdelsamp, flt.groupdelsamp + n_rgb, flt.downfact)
    # perform range multi-rate filtering in frequency domain
    with fft.set_workers(n_cpu):
        echo_flt_fft = fft.fft(echo_up, n=nfft, axis=1) * coefs_fft
        echo_flt = fft.ifft(echo_flt_fft, axis=1)[:, slice_rg]
        echo_flt = echo_flt.astype(echo.dtype, copy=False)
    return echo_flt


def _join_paths(path1: str, path2: str) -> str:
    """Join two paths to be used in HDF5"""
    sep = '/'
    if path1.endswith(sep):
        sep = ''
    return path1 + sep + path2


def copy_swath_except_echo_sr_h5(fid_in, fid_out, swath_path,
                                 frq_pol, freq_band_in='A', n_sbsw=1):
    """
    Copy all groups and datasets under swath from input HDF5 to output
    HDF5 excpet for echo products and slant range vector.
    Slant range and echo fileds are skipped because those need to be
    updated with new size and shape after filtering in range direction.
    Given output may have two frequency bands as opposed to the input, e.g.,
    in QQ case, the input values are repeated for output products.

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
        polarization as values for output products!
        This could be both "A" and "B", SSP, for QQ case of DM1.
    freq_band_in : str, default='A'
        Frequency band char for the input which shall be either
        "A" or "B" for DM1 not both!
    n_sbsw : int, default=1
        Number of valid subswath with valid values for the output L0B.

    """
    # form input single freq-band path.
    band_path_in = _join_paths(swath_path, f'frequency{freq_band_in}/')
    for freq_band in frq_pol:
        # build band path
        band_path = _join_paths(swath_path, f'frequency{freq_band}/')
        # create freq band group for output product
        grp_band = fid_out.require_group(band_path)
        # list of all TxRx products
        txrx_pols = frq_pol[freq_band]
        # list of group names txp where "p" is all TX pols
        # TX is either H or V per output freq_band!
        txp = {f'tx{p[0]}' for p in txrx_pols}
        # list of group names rxp where "p" is all RX pols
        # RX can be both H and V per freq_band
        rxp = {f'rx{p[1]}' for p in txrx_pols}
        for band_item in fid_in[band_path_in]:
            if band_item in txp:
                # form TX path
                tx_path = band_path + band_item + '/'
                tx_path_in = band_path_in + band_item + '/'
                # create TX group for output product
                grp_tx = fid_out.require_group(tx_path)
                for tx_item in fid_in[tx_path_in]:
                    if tx_item in rxp:
                        rx_path_in = tx_path_in + tx_item + '/'
                        # loop over all output RXs
                        for rx in rxp:
                            # form RX path
                            rx_path = tx_path + rx + '/'
                            # create RX group for output product
                            grp_rx = fid_out.require_group(rx_path)
                            for rx_item in fid_in[rx_path_in]:
                                if rx_item not in txrx_pols:
                                    # copy all RX fields except echo product
                                    fid_in.copy(rx_path_in + rx_item, grp_rx)
                    elif tx_item == 'slantRange':
                        continue

                    elif 'validSamplesSubSwath' in tx_item:
                        # simply copy good subswaths
                        if int(tx_item[-1]) <= n_sbsw:
                            fid_in.copy(tx_path_in + tx_item, grp_tx)

                    else:  # copy the rest of TX as it is
                        fid_in.copy(tx_path_in + tx_item, grp_tx)
                        # update number of good subswath
                        if tx_item == 'numberOfSubSwaths':
                            grp_tx[tx_item][()] = n_sbsw

            else:
                if band_item[:2] != 'tx':
                    # copy the rest of frequency band as it is
                    # as long as it is not another TX pol given the
                    # txp will be stored in another frequency band
                    fid_in.copy(band_path_in + band_item, grp_band)


def copy_update_identification_h5(
        fid_in, fid_out, ident_path, list_freqs):
    """
    Copy all items of identification group from input file and update
    a couple of items for the output product such as DM flag and
    list of frequencies.

    Parameters
    ----------
    fid_in : h5py.File
        File-like object for input HDF5 L0B product
    fid_out : h5py.File
        File-like object for output HDF5 L0B product.
    ident_path : str
        HDF5 path for identification.
    list_freqs : list(str)
        List of frequency band chars.

    """
    grp_ident = fid_out.require_group(ident_path)
    for item in fid_in[ident_path]:
        p = _join_paths(ident_path, item)
        if item == 'listOfFrequencies':
            ds = grp_ident.create_dataset(
                name=item, data=np.bytes_(list_freqs))
            # copy attributes
            for a_name, a_val in fid_in[p].attrs.items():
                ds.attrs[a_name] = a_val
        else:  # copy everything else
            fid_in.copy(p, grp_ident)
    # fix DM flag
    grp_ident['diagnosticModeFlag'][()] = np.uint8(0)


def create_echo_dataset_h5(fid_in, fid_out, band_path, txrx_pol,
                           prod_shape, prod_dtype, comp_level_h5,
                           band_path_in):
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
        Frequency band path in L0B HDF5 output product.
    txrx_pol : str
        TxRx polarization of the output echo product
    prod_shape: tuple(int, int)
        Shape of the new echo product
    prod_dtype: np.dtype
        Date type of the new echo product
    comp_level_h5 : int
        Compression level of gzip in HDF5 used for echo raster.
    band_path_in : str
        HDF5 path for frequency band of the input which shall contains
        either frequency band "A" or "B" for DM1 not both!

    Returns
    -------
    h5py.Dateset
        HDF5 Dataset for the output echo product

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
    rx_path_in = _join_paths(
        band_path_in, f'tx{txrx_pol[1]}/rx{txrx_pol[1]}/')
    p_path_in = rx_path_in + 2 * txrx_pol[1]
    for a_name, a_val in fid_in[p_path_in].attrs.items():
        dset_prod.attrs[a_name] = a_val
    return dset_prod


def create_slantrange_update_range_h5(
        fid_in, fid_out, tx_grp, sr_out, ground_rg_space,
        tx_grp_in):
    """
    Create slant range vector and update range-related fields in
    the output HDF5 product.

    Parameters
    ----------
    fid_in : h5py.File
        File-like object for input HDF5 L0B product
    fid_out : h5py.File
        File-like object for output HDF5 L0B product
    tx_grp : str
        Path for TX group in output L0B HDF5 product.
    sr_out : isce3.core.Linspace
        Slant range vector for the output product
    ground_rg_space : float
        Ground range spacing for the output product
    tx_grp_in : str
        Path for TX group in input L0B HDF5 product.

    """
    dset_sr = fid_out[tx_grp].require_dataset(
        'slantRange', shape=sr_out.shape, dtype=sr_out.dtype,
        data=np.asarray(sr_out), exact=True)
    # copy attributes from input HDF5 as it is
    sr_path = _join_paths(tx_grp_in, 'slantRange')
    for a_name, a_val in fid_in[sr_path].attrs.items():
        dset_sr.attrs[a_name] = a_val
    # update already-existed range-related fields under tx
    for item, val in zip(
        ['slantRangeSpacing', 'sceneCenterGroundRangeSpacing'],
            [sr_out.spacing, ground_rg_space]):
        item_path = _join_paths(tx_grp, item)
        fid_out[item_path][()] = val


def update_subswath_h5(fid_out, sbsw_path_prefix, sbswath):
    """
    Update valid subswath values for an exisiting fields in new hdf5
    product.

    Parameters
    ----------
    fid_out : h5py.File
        File-like object for output HDF5 L0B product
    sbsw_path_prefix : str
        Path prefix for all validsubswath datasets in L0B HDF5 product.
    sbswath : np.ndarray(int)
        3-D array for all valid subswaths to be used for the output product.

    """
    num_sbsw, _, _ = sbswath.shape
    for n in range(num_sbsw):
        sbsw_path = sbsw_path_prefix + f'{n + 1}'
        fid_out[sbsw_path][...] = sbswath[n]


def form_products_dm1(raw: Raw):
    """Form full quasi-quad science products for DM1 L0B product.
    In the nominal operation, where there are two co-pols `HH` and `VV`
    sitting on different carries frequencies (quasi mode), total four
    products co-pol and cross-pol under two frequency bands `A` and `B`
    will be generated. Something like {'A':['HH', 'HV'], 'B':['VV', 'VH']}.
    If one of the polrization is not available in DM1 L0B, then simply a
    Dual polarization product will be generated under frequency band `A`.
    Something like {'A':['HH', 'HV']}.

    Parameters
    ----------
    raw : nisar.products.readers.Raw
        DM1 L0B Raw product

    Returns
    -------
    dict
        Product dictionary {freq_band_str : list(txrx_pols)}

    """
    frq = raw.frequencies
    if len(frq) != 1:  # 'A' or 'B', not both!
        # len(frq) > 1 -> split spectrum
        raise ValueError(f'Split spectrum {frq} for non-filtered DM1!')
    f = frq[0]
    tx_p = [pol[0] for pol in raw.polarizations[f]]
    if len(tx_p) == 1:  # HH or VV (one may be missing!)
        warn('QQ expected for DM1 per mode table but got single'
             f' TX pol {tx_p[0]}!')
        frq_pol_out = raw.polarizations
    else:  # two TX pols, HH and VV
        fc_h = raw.getCenterFrequency(f, 'H')
        fc_v = raw.getCenterFrequency(f, 'V')
        # check the order of freq band for H and V
        if np.isclose(fc_h, fc_v):
            raise ValueError(
                f'TX H freq {fc_h} (MHz) is the same as V {fc_v} (MHz)!')
        elif fc_h < fc_v:
            frq_pol_out = dict(
                A=['HH', 'HV'],
                B=['VV', 'VH']
            )
        else:  # fc_h > fc_v:
            frq_pol_out = dict(
                A=['VV', 'VH'],
                B=['HH', 'HV']
            )
    return frq_pol_out


def number_valid_subswath(sbsw):
    """
    Get number of valid subswath with unique (start, stop) from
    sub-swath array.

    Parameters
    ----------
    sbsw : np.ndarray(int)
        3-D array of (start, stop) range bins representing valid
        sub-swath per range line. The shape is
        (number of sub-swaths, number of range lines, 2)

    Returns
    -------
    int
        Number of valid subswaths with valid values.

    """
    num = 0
    for sb in sbsw:
        if sb.min() == sb.max():
            break
        else:
            num += 1
    if num == 0:
        raise ValueError('There is no valid subswath with valid values!')
    return num


def nisar_l0b_dm1_to_science(args):
    """Create NISAR science L0B from DM1 L0B product"""

    tic = time.time()
    # set logger
    logger = set_logger('SCIENCE-L0B-FROM-DM1')

    # oversampling factor in range
    ovsf_rg = args.ovsf_rg
    if ovsf_rg is not None and ovsf_rg <= 1:
        raise ValueError('OVSF in range must be > 1!')

    plot = args.plot
    if plot and plt is None:
        logger.warning('No plot! Missing "matplotlib"!')
        plot = False

    # set number of CPU
    if args.num_cpu is None:
        n_cpu = os.cpu_count()
    elif args.num_cpu < 1:
        raise ValueError(f'num_cpu must be >= 1, got {args.num_cpu}')
    else:
        n_cpu = min(os.cpu_count(), args.num_cpu)
    logger.info(f'Number of CPUs used in FFT interpolation -> {n_cpu}')

    # parse raw
    fid_in = h5py.File(args.l0b_file, 'r', swmr=True)
    raw = Raw(hdf5file=args.l0b_file)

    # check DM flag to make sure it is DM1
    dm_flag = raw.identification.diagnosticModeFlag
    if dm_flag != 1:
        raise ValueError('The is not a DM1 product! DM flag -> {dm_flag}!')

    # form ordered output products for DM1 QQ
    frq_pol_out = form_products_dm1(raw)
    logger.info(f'List of final output products -> {frq_pol_out}')

    # get all products
    frq_pol = raw.polarizations
    # first frequency band and pol
    freq_band_in = list(frq_pol)[0]
    txrx_pol = frq_pol[freq_band_in][0]
    # input HDF5 single-freq band path
    band_path_in = raw.BandPath(freq_band_in)

    # get ref epoch and build AZ slice generator
    epoch, azt_raw = raw.getPulseTimes(freq_band_in, txrx_pol[0])
    n_rgl_tot = azt_raw.size

    # form BFPQ LUT for uniform quantizer to be used for output product
    if args.sign_mag:
        logger.info(
            'Assumed signed magnitude representation of echo data!')
    else:
        logger.info(
            'Assumed twos complement representation of echo data!')
    # BFPQ decoder used for parsing input raw echo
    bfpq_uq = build_uniform_quantizer_lut_l0b(
        args.nbits, twos_complement=not args.sign_mag)
    # BFPQ decoder for basebanded requantized 16-bit output raw echo
    bfpq_uq_16bit = build_uniform_quantizer_lut_l0b(16)

    # get in/out files and file objects
    p_in = Path(args.l0b_file)
    out_path = Path(args.out_path)
    if args.prod_name is None:
        suffix = '_Science.h5'
        file_out = p_in.stem + suffix
    else:
        file_out = args.prod_name
    logger.info(f'Filename of output Science product -> "{file_out}"')
    fid_out = h5py.File(file_out, 'w')

    # copy the entire data for metadata, low/high res telemetry
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
    # and update list of frequencies
    copy_update_identification_h5(
        fid_in, fid_out, raw.IdentificationPath, list(frq_pol_out))

    # get number of valid subswath
    sbswath = raw.getSubSwaths(freq_band_in, txrx_pol[0])
    logger.info(
        f'Number of subswath in the input file -> {sbswath.shape[0]}'
    )
    n_sbsw = number_valid_subswath(sbswath)
    logger.info(
        f'Number of subswath in the output file -> {n_sbsw}'
    )
    # copy entire swath except for TxRx echo products and slant range
    copy_swath_except_echo_sr_h5(
        fid_in, fid_out, raw.SwathPath, frq_pol_out, freq_band_in, n_sbsw)

    # loop over all products and bands
    # form a vector of range line slices used for all products
    rgl_slices = list(slice_gen(n_rgl_tot, args.num_rgl))
    logger.info(f'Number of AZ blocks -> {len(rgl_slices)}')

    for freq_band in frq_pol_out:
        # group path for frequency band
        band_path = raw.BandPath(freq_band) + '/'

        for txrx_pol in frq_pol_out[freq_band]:
            logger.info(
                f'Processing frequency band {freq_band} and Pol '
                f'{txrx_pol} ...')

            # product group path
            tx_grp = band_path + f'tx{txrx_pol[0]}/'
            prod_grp = tx_grp + f'rx{txrx_pol[1]}/'

            # get chirp parameters
            fc, fs, rate, pw = raw.getChirpParameters(
                freq_band_in, txrx_pol[0])
            # get chirp BW and final output sampling rate
            bw = abs(rate * pw)
            # set over sampling factor if None
            # based on NISAR mode
            if ovsf_rg is None:
                # check if it is 80 MHz mode with BW=77MHz
                # then use a slightly higher factor 96MHz/77MHz
                if np.isclose(bw, 77e6) and not (fs < 96e6):
                    ovsf_rg = 96e6 / bw
                # otherwise, set it to 1.2
                else:
                    ovsf_rg = 1.2
            logger.info(
                f'Over-sampling factor in range is set to -> {ovsf_rg}'
            )
            # build multi-rate BPF FIR filter object
            flt = build_multi_rate_fir_filter(fs, bw, ovsf_rg, fc)
            logger.info(
                'Over-sampling factor used in the FIR filter -> '
                f'{flt.rateout / bw}')
            logger.info(f'Number of filter taps -> {flt.numtaps}')
            logger.info('(up, down) sampling factors -> '
                        f'({flt.upfact}, {flt.downfact})')
            logger.info(
                'Output sampling rate after multi-rate filtering (MHz) -> '
                f'{flt.rateout * 1e-6:.3f}')

            # parse echo based on RX pol with co-pol value!
            dset = raw.getRawDataset(freq_band_in, 2 * txrx_pol[1])

            # check if BFPQ LUT is the correct one, if not replace it
            if not np.all(np.isclose(dset.table, bfpq_uq)):
                logger.warning(
                    f'BFPQLUT is not {args.nbits}-bit uniform quantizer!'
                    'It will be replaced!')
                dset.table[...] = bfpq_uq

            nazb, nrgb_in = dset.shape
            nrgb_up = nrgb_in * flt.upfact
            nrgb_out = round(nrgb_up / flt.downfact)
            logger.info(f'Number of output complex samples -> {nrgb_out}')

            # build new slant range vector
            sr = raw.getRanges(freq_band_in, txrx_pol[0])
            sr_space_out = speed_of_light / (2 * flt.rateout)
            sr_out = Linspace(sr.first, sr_space_out, nrgb_out)
            logger.info(
                f'Output slant range spacing (m) -> {sr_out.spacing:.3f}'
            )
            # dump slant range vectors and modify already-existed
            # slant-range-related fields
            tx_grp_in = band_path_in + f'/tx{txrx_pol[0]}/'
            rng_space_scalar = sr_out.spacing / sr.spacing
            ground_rg_space = (
                rng_space_scalar *
                fid_in[tx_grp_in]['sceneCenterGroundRangeSpacing'][()]
            )
            logger.info(
                'Updated ground range spacing at the center of the scene '
                f'(m) -> {ground_rg_space:.3f}')
            create_slantrange_update_range_h5(
                fid_in, fid_out, tx_grp, sr_out, ground_rg_space,
                tx_grp_in)

            # take fft of filter coefs at flt.ratein (upsampling rate)!
            nfft_rg = fft.next_fast_len(flt.numtaps + nrgb_up)
            logger.info(f'NFFT for range filtering -> {nfft_rg}')
            coefs_flt_fft = fft.fft(flt.coefs, n=nfft_rg)

            # down conversion vector w/o taking into account the phase
            # offset due to first-sample absolute round-trip time or data
            # window position (DWP).
            downconv = np.exp(
                -1j * 2 * np.pi * fc / flt.rateout * np.arange(nrgb_out),
                dtype=dset.dtype)

            # function to scale range bins for valid subswath
            rgb_scalar = 1 / rng_space_scalar
            def scale_rgb(b): return np.int_(np.round(b * rgb_scalar))
            # adjust the range bins in valid sub swath array
            sbswath = raw.getSubSwaths(freq_band_in, txrx_pol[0])[:n_sbsw]
            sbswath[...] = scale_rgb(sbswath)
            sbswath[sbswath > nrgb_out] = nrgb_out
            # update valid subswaths values in L0B
            sbsw_path_prefix = tx_grp + 'validSamplesSubSwath'
            update_subswath_h5(fid_out, sbsw_path_prefix, sbswath)

            # create a placeholder for echo product and use uniform-quantizer
            # BFPQ LUT in place of BFPQ ones.
            dset_prod_out = create_echo_dataset_h5(
                fid_in, fid_out, band_path, txrx_pol, (nazb, nrgb_out),
                dset.dtype_storage, args.comp_level_h5, band_path_in)

            bfpq_path = prod_grp + 'BFPQLUT'
            fid_out[bfpq_path][:] = bfpq_uq_16bit

            # loop over AZ blocks
            for n_blk, rgl_slice in enumerate(rgl_slices, start=1):
                logger.info(f'Processing AZ block # {n_blk} ...')

                # perform filtering in range per AZ block
                echo_flt = range_filtering_per_az_block(
                    dset[rgl_slice], coefs_flt_fft, flt, n_cpu
                )[:, :nrgb_out]
                # down convert decimated block
                echo_flt *= downconv
                # plot float-point DBFed raster per AZ block
                if plot:
                    plt.figure(figsize=(8, 6))
                    plt.imshow(abs(echo_flt), cmap='grey', aspect='auto')
                    plt.xlabel('Range Bins (-)')
                    plt.ylabel('Azimuth Bins (-)')
                    plt.title(f'Science Echo for AZ Block # {n_blk}')
                    plot_name = out_path.joinpath(
                        f'Plot_Science_Echo_Freq{freq_band}_'
                        f'Pol{txrx_pol}_AzBlock{n_blk}.png'
                    )
                    logger.info('Filename of Science Raster Plot -> '
                                f'{plot_name}')
                    plt.savefig(plot_name)
                    plt.close()
                # no need to check for overflow given the entire filtering
                # and down conversion has zero dB gain!
                # quantize and recast
                dset_prod_out.write_direct(
                    echo_flt.view('f4').round().astype('uint16').view(
                        dset.dtype_storage), dest_sel=rgl_slice)

    # close in/out HDF5 files
    fid_in.close()
    fid_out.close()

    logger.info(f'Elapsed time (sec) -> {time.time() - tic:.1f}')


if __name__ == '__main__':
    nisar_l0b_dm1_to_science(cmd_line_parser())
