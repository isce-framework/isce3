#!/usr/bin/env python3
"""
Generate Doppler Centroid product from L0B data
"""
import os
import time
import argparse as argp
import numpy as np
from datetime import datetime, timezone

from nisar.pointing import doppler_lut_from_raw
from nisar.log import set_logger
from nisar.products.readers.Raw import Raw
from isce3.core import TimeDelta, Linspace
from nisar.products.readers.antenna import AntennaParser
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.products.readers.attitude import load_attitude_from_xml
from isce3.io import Raster
from isce3.geometry import DEMInterpolator
from nisar.workflows.gen_el_null_range_product import (
    copol_or_desired_product_from_raw)


def cmd_line_parser():
    """Parse command line input arguments.

    Notes
    -----
    It also allows parsing arguments via an ASCII file
    by using prefix char "@".

    Returns
    -------
    argparse.Namespace

    """
    prs = argp.ArgumentParser(
        description='Estimate Doppler centroid from L0B raw echo and creates '
        'a 2-D Doppler LUT dumped into a CSV file',
        fromfile_prefix_chars="@",
        formatter_class=argp.ArgumentDefaultsHelpFormatter
    )
    prs.add_argument('filename_l0b', type=str,
                     help='Filename of HDF5 L0B product')
    prs.add_argument('--ant', type=str, dest='antenna_file',
                     help='Filename of HDF5 Antenna product used to extract '
                     'averaged azimuth angle for EL cuts of TX + RX pol of '
                     'first beam. It also uses for Doppler ambiguity '
                     'calculation. If not provided, the azimuth angle is '
                     'assumed to be zero. This is required for multi-channel'
                     ' L0B product (NISAR DM2)!')
    prs.add_argument('--dem', type=str, dest='dem_file',
                     help='DEM raster file in (GDAL-compatible format such'
                     ' as GeoTIFF) containing heights w.r.t. WGS-84 ellipsoid.'
                     ' Default is constant height set by "ref_height".')
    prs.add_argument('--ref_height', type=float, dest='ref_height', default=0,
                     help='Reference height in (m) w.r.t WGS84 ellipsoid.'
                     ' It will be simply used if "dem" file is not provided')
    prs.add_argument('-f', '--freq', type=str, choices=['A', 'B'],
                     dest='freq_band',
                     help='Frequency band such as "A". If set, the products '
                     'over desired `txrx_pol` or over all co-pols will be '
                     'processed. Otherwise, either all frequency bands with '
                     'desired `txrx_pol` or with all co-pols will be '
                     'processed (default)!')
    prs.add_argument('-p', '--pol', type=str, dest='txrx_pol',
                     choices=['HH', 'VV', 'HV', 'VH', 'RH', 'RV', 'LH', 'LV'],
                     help=('TxRx Polarization such as "HH". If set, the '
                           'products either over specified `freq_band` or over'
                           ' all available bands will be processed. Otherwise,'
                           ' all co-pols in each requested frequency band will'
                           ' be processed (default)!')
                     )
    prs.add_argument('-r', '--rgb', type=int, dest='num_rgb_avg', default=8,
                     help='Number of range bins to be averaged in Doppler '
                     'Estimator block. Shall be equal or larger than 1.')
    prs.add_argument('-a', '--az_block_dur', type=float, dest='az_block_dur',
                     default=4.0,
                     help='Azimuth block duration in seconds defining time-'
                     'domain correlator length used in Doppler estimator.')
    prs.add_argument('-t', '--time_interval', type=float, dest='time_interval',
                     default=2.0,
                     help='Time stamp interval between azimuth blocks in '
                     'seconds. Must not be larger than "az_block_dur".')
    prs.add_argument('-m', '--method', type=str, dest='dop_method',
                     default='CDE', choices=['SDE', 'CDE'],
                     help='Time-domain Doppler estimator methods "CDE"/"SDE"'
                     ' which are Correlator/Sign Doppler Estimator.')
    prs.add_argument('--subband', action='store_true', dest='subband',
                     help='Perform fast-time frequency subbanding on top of '
                     'time-domain correlator in Doppler estimator')
    prs.add_argument('-d', '--deg', type=int, dest='polyfit_deg',
                     default=3, help='Degree of the polyfit.')
    prs.add_argument('--polyfit', action='store_true', dest='polyfit',
                     help='If set, it will replace actual estimated doppler '
                     'by its polyfitted ones in slant range.')
    prs.add_argument('--plot', action='store_true', dest='plot',
                     help='Plot Doppler centroids and save them in '
                     '*.png files at the specified output path')
    prs.add_argument('-o', '--out', type=str, dest='out_path', default='.',
                     help='Output directory to dump Doppler product as well as'
                     'PNG plots.')
    prs.add_argument('--orbit', type=str, dest='orbit_file',
                     help='Filename of an external orbit XML file. The orbit '
                     'data will be used in place of those in L0B. Default is '
                     'orbit data stored in L0B.')
    prs.add_argument('--attitude', type=str, dest='attitude_file',
                     help='Filename of an external attitude XML '
                     'file. The attitude data will be used in place of those '
                     'in L0B. Default is attitude data stored in L0B.')

    return prs.parse_args()


def gen_doppler_range_product(args):
    """Generate Doppler-Range LUT Product.

    It generates Doppler centroid LUT as a function of slant range
    at various azimuth/pulse times and dump them into a CSV file.

    The format of the file and output filename convention is defined
    in reference [1]_.

    Parameters
    ----------
    args : argparse.Namespace
        All input arguments parsed from a command line or an ASCII file.

    References
    ----------
    .. [1] D. Kannapan, "D&C Radar Data Product SIS," JPL D-104976,
        December 3, 2020.

    """
    # Const
    PREFIX_NAME_CSV = 'NISAR_ANC'

    tic = time.time()
    # set logger
    logger = set_logger("DopplerRangeProduct")

    # get Raw object
    raw_obj = Raw(hdf5file=args.filename_l0b)
    # get the SAR band char
    sar_band_char = raw_obj.sarBand
    logger.info(f'SAR band char -> {sar_band_char}')

    # logic for frequency band and TxRx polarization choices.
    # form a new dict "frq_pol" with key=freq_band and value=[txrx_pol]
    frq_pol = copol_or_desired_product_from_raw(
        raw_obj, args.freq_band, args.txrx_pol)
    logger.info(f'List of selected frequency bands and TxRx Pols -> {frq_pol}')

    # operation mode, whether DBF (single or a composite channel) or
    # 'DM2' (multi-channel)
    # TODO: the updated "diagnosticModeFlag" under "identification" in
    # L0B product shall be used to tell us about "OP_MODE".
    # For DM2, its value is 2. For DBF or single-channel SAR like
    # ALOS1 PALSAR its value is 0.  To avoid failure for older L0B
    # products, we use number of dimension of raw dataset to decide
    # about "OP_MODE" for now!
    freq_band = list(frq_pol)[0]
    txrx_pol = frq_pol[freq_band][0]
    raw_dset = raw_obj.getRawDataset(freq_band, txrx_pol)
    if raw_dset.ndim == 3:
        op_mode = 'DM2'
    else:  # single channel SAR
        op_mode = 'DBF'
    del raw_dset

    # build orbit and attitude object if external files are provided
    if args.orbit_file is None:
        orbit = None
    else:
        logger.info('Parsing external orbit XML file')
        orbit = load_orbit_from_xml(args.orbit_file)

    if args.attitude_file is None:
        attitude = None
    else:
        logger.info('Parsing external attitude XML file')
        attitude = load_attitude_from_xml(args.attitude_file)

    # build antenna object if provided
    if args.antenna_file is None:
        if op_mode == 'DM2':
            raise ValueError('Multi-channel (DM2) L0B requires Antenna file!')
        ant = None
    else:
        ant = AntennaParser(args.antenna_file)

    # build DEM object
    if args.dem_file is None:
        dem = DEMInterpolator(args.ref_height)
    else:
        dem = DEMInterpolator(Raster(args.dem_file))

    # get keyword args for function "doppler_lut_from_raw"
    kwargs = {key: val for key, val in vars(args).items() if
              'file' not in key and key not in [
                  'ref_height', 'freq_band', 'txrx_pol']}

    # loop over all desired frequency bands and their respective desired
    # polarizations
    for freq_band in frq_pol:
        for txrx_pol in frq_pol[freq_band]:
            # generate Doppler LUT2d from Raw L0B
            dop_lut, ref_utc, mask_rgb, corr_coef, txrx_pol, centerfreq, _ = \
                doppler_lut_from_raw(raw_obj, orbit=orbit, attitude=attitude,
                                     ant=ant, dem=dem, logger=logger,
                                     freq_band=freq_band, txrx_pol=txrx_pol,
                                     **kwargs)

            # check out antenna object to extract azimuth angle for EL cuts
            # used for Doppler CSV product
            if ant is None:
                az_ang_deg = 0.0
                logger.warning(
                    'No antenna file! Azimuth angle for Doppler product is '
                    'assumed to be zero!'
                )
            else:
                logger.info(
                    'Extracting the azimuth angle of EL cuts from antenna '
                    'file.'
                )
                ant_rx = ant.el_cut(pol=txrx_pol[1])
                az_ang = ant_rx.cut_angle
                # if TX pol is different from RX pol then take average of both
                if txrx_pol[0] != txrx_pol[1]:
                    # To cover TX L/R circular pol, the following condition
                    # is needed
                    if txrx_pol[1] == 'H':
                        tx_pol = 'V'
                    else:
                        tx_pol = 'H'
                    ant_tx = ant.el_cut(pol=tx_pol)
                    az_ang += ant_tx.cut_angle
                    az_ang *= 0.5
                az_ang_deg = np.rad2deg(az_ang)
                logger.info(
                    'Azimuth angle extracted from antenna file -> '
                    f'{az_ang_deg:.3f} (deg)'
                )

            # form Linspace object for uniformly-spaced azimuth time and
            # slant range
            azt_lsp = Linspace(
                dop_lut.y_start, dop_lut.y_spacing, dop_lut.length
            )
            sr_lsp = Linspace(
                dop_lut.x_start, dop_lut.x_spacing, dop_lut.width
            )

            # get the first and last utc azimuth time w/o fractional seconds
            # in "%Y%m%dT%H%M%S" format to be used as part of CSV product
            # filename.
            dt_utc_start = sec2str(ref_utc, azt_lsp.first)
            dt_utc_stop = sec2str(ref_utc, azt_lsp.last)
            # get current time w/o fractional seconds in "%Y%m%dT%H%M%S" format
            # used as part of CSV product filename
            dt_utc_cur = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')

            # naming convention of CSV file and product spec is defined in Doc:
            # See reference [1]
            name_csv = (
                f'{PREFIX_NAME_CSV}_{sar_band_char}{freq_band}_{op_mode}_'
                f'DOPP_{txrx_pol}_{dt_utc_cur}_{dt_utc_start}_{dt_utc_stop}'
                '.csv'
            )
            file_csv = os.path.join(args.out_path, name_csv)
            logger.info(
                f'Dump Doppler product in "CSV" format to file -> {file_csv}'
            )

            with open(file_csv, 'wt') as fid_csv:
                fid_csv.write(
                    'UTC Time,Frequency (Hz),Doppler (Hz),Range (m),Azimuth'
                    ' (deg),Correlation\n'
                )
                # loop over azimuth time and slant ranges
                for i_row, azt in enumerate(azt_lsp):
                    tm_utc_str = sec2isofmt(ref_utc, azt)

                    for i_col, sr in enumerate(sr_lsp):
                        fid_csv.write(
                            '{:s},{:.1f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
                                tm_utc_str, centerfreq,
                                dop_lut.data[i_row, i_col],
                                sr, az_ang_deg,
                                (mask_rgb[i_row, i_col] *
                                 corr_coef[i_row, i_col])
                            )
                        )

    # total elapsed time
    logger.info(f'Elapsed time -> {time.time() - tic:.1f} (sec)')


def sec2isofmt(ref_utc: 'isce3.core.DateTime', seconds: float) -> str:
    """seconds to isoformat string"""
    return (ref_utc + TimeDelta(seconds)).isoformat_usec()


def sec2str(ref_utc: 'isce3.core.DateTime', seconds: float) -> str:
    """seconds to string format '%Y%m%dT%H%M%S'"""
    fmt = '%Y%m%dT%H%M%S'
    dt_iso = sec2isofmt(ref_utc, seconds)
    return datetime.fromisoformat(dt_iso.split('.')[0]).strftime(fmt)


if __name__ == "__main__":
    """Main driver"""
    gen_doppler_range_product(cmd_line_parser())
