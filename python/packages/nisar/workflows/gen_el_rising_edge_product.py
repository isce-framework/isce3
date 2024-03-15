#!/usr/bin/env python3
"""
Generate El rising edge product from either DBFed SwwepSAR L0B (science mode)
or a single-channel SAR over homogenous scene plus Antenna HDF5 plus
[DEM raster]. The output will be used for pointing analyses by D&C team.
The product spec is the same as that of `el_null_range`.
"""
import os
import time
import argparse as argp
import numpy as np
from datetime import datetime

from nisar.products.readers.Raw import Raw
from nisar.products.readers.antenna import AntennaParser
from isce3.geometry import DEMInterpolator
from isce3.io import Raster
from nisar.pointing import el_rising_edge_from_raw_ant
from nisar.log import set_logger
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.products.readers.attitude import load_attitude_from_xml
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
        description=('Generate EL rising-edge-range product from L0B (science)'
                     ' + Antenna HDF5 + [DEM raster] data'),
        formatter_class=argp.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@"
    )
    prs.add_argument('--l0b', type=str, required=True, dest='l0b_file',
                     help='Filename of HDF5 L0B science product')
    prs.add_argument('--ant', type=str, required=True, dest='antenna_file',
                     help='Filename of HDF5 Antenna product')
    prs.add_argument('-b', '--beam_num', type=int, default=1, dest='beam_num',
                     help='Beam number to pick the antenna pattern from the'
                     ' antenna HDF5 file. This is only used for single-channel'
                     ' SAR (not multi-channel DBFed SweepSAR!)')
    prs.add_argument('-d', '--dem_file', type=str, dest='dem_file',
                     help='DEM raster file in (GDAL-compatible format such as '
                     'GeoTIFF) containing heights w.r.t. WGS-84 ellipsoid. '
                     'Default is no DEM!')
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
                           ' the first co-pol of the specified `freq_band` or '
                           'all co-pols over all frequency bands will be '
                           'processed (default)!')
                     )
    prs.add_argument('--orbit', type=str, dest='orbit_file',
                     help='Filename of an external orbit XML file. The orbit '
                     'data will be used in place of those in L0B. Default is '
                     'orbit data stored in L0B.')
    prs.add_argument('--attitude', type=str, dest='attitude_file',
                     help='Filename of an external attitude XML '
                     'file. The attitude data will be used in place of those '
                     'in L0B. Default is attitude data stored in L0B.')
    prs.add_argument('-a', '--az_block_dur', type=float, dest='az_block_dur',
                     default=3.0, help='Duration of azimuth block in seconds.'
                     ' This value will be limited by total azimuth duration.'
                     ' The value must be equal or larger than the mean PRI.')
    prs.add_argument('-o', '--out', type=str, dest='out_path', default='.',
                     help='Output directory to dump PNG files if `--plot` and'
                     'the EL rising-edge product. The product is CSV file '
                     'whose name conforms to JPL D-104976.')
    prs.add_argument('--no_dbf_norm', action='store_true',
                     dest='no_dbf_norm', help='Do not apply power norm to DBF')
    prs.add_argument('--plot', action='store_true', dest='plot',
                     help='Generates PNG plots and dump')
    prs.add_argument('-r', '--ref_height', type=float, dest='ref_height',
                     default=0.0,
                     help=('Reference height in (m) w.r.t WGS84 ellipsoid. It '
                           'will be simply used if "dem_file" is not provided')
                     )
    prs.add_argument('--no_weight', action='store_true', dest='no_weight',
                     help='Do not apply SNR-based weights to the cost function'
                     )
    return prs.parse_args()


def gen_el_rising_edge_product(args):
    """Generate EL Rising-edge Product.

    It generates rising edge location in elevation (EL) direction as a
    function slant range at various azimuth/pulse times and dump them
    into a CSV file.

    Parameters
    ----------
    args : argparse.Namespace
        All input arguments parsed from a command line or an ASCII file.

    Notes
    -----
    The format of the file and output filename convention is defined
    in reference [1]_.

    References
    ----------
    .. [1] D. Kannapan, "D&C Radar Data Product SIS," JPL D-104976,
        December 3, 2020.

    """
    # Const
    prefix_name_csv = 'NISAR_ANC'
    # operation mode, whether DBF (single or a composite channel) or
    # 'DM2' (multi-channel). Null only possible via 'DM2'!
    # Rising edge is only possible via [DBFed] science mode.
    op_mode = 'DBF'

    tic = time.time()
    # set logging
    logger = set_logger("ELRisingEdgeProduct")

    # build Raw object
    raw = Raw(hdf5file=args.l0b_file)
    # get the SAR band char
    sar_band_char = raw.sarBand
    logger.info(f'SAR band char -> {sar_band_char}')

    # build antenna object from antenna file
    ant = AntennaParser(args.antenna_file)

    # build dem interp object from DEM raster or ref height
    if args.dem_file is None:  # set to a fixed height
        dem_interp = DEMInterpolator(args.ref_height)
    else:  # build from DEM Raster file
        dem_interp = DEMInterpolator(Raster(args.dem_file))

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

    # logic for frequency band and TxRx polarization choices.
    # form a new dict "frq_pol" with key=freq_band and value=[txrx_pol]
    frq_pol = copol_or_desired_product_from_raw(
        raw, freq_band=args.freq_band, txrx_pol=args.txrx_pol)
    logger.info(f'List of selected frequency bands and TxRx Pols -> {frq_pol}')

    # get keyword args for function "el_rising_edge_from_raw_ant"
    kwargs = {key: val for key, val in vars(args).items() if
              key in ['az_block_dur', 'beam_num', 'out_path',
                      'plot']}

    # loop over all desired frequency bands and their respective desired
    # polarizations
    for freq_band in frq_pol:
        for txrx_pol in frq_pol[freq_band]:
            # EL rising edge process
            (el_ofs, az_dtm, el_fl, sr_fl, lka_fl, msk, cvg, pf_echo,
             pf_ant, pf_wgt) = el_rising_edge_from_raw_ant(
                 raw, ant, dem_interp=dem_interp, orbit=orbit,
                 attitude=attitude, logger=logger,
                 dbf_pow_norm=not args.no_dbf_norm,
                 apply_weight=not args.no_weight, freq_band=freq_band,
                 txrx_pol=txrx_pol, **kwargs
                 )
            # get the first and last utc azimuth time w/o fractional seconds
            # in "%Y%m%dT%H%M%S" format to be used as part of CSV product
            # filename.
            dt_utc_first = dt2str(az_dtm[0])
            dt_utc_last = dt2str(az_dtm[-1])
            # get current time w/o fractional seconds in "%Y%m%dT%H%M%S" format
            # used as part of CSV product filename
            dt_utc_cur = datetime.now().strftime('%Y%m%dT%H%M%S')

            # naming convention of CSV file and product spec is defined in Doc:
            # See reference [1]
            name_csv = (
                f'{prefix_name_csv}_{sar_band_char}{freq_band}_{op_mode}_NULL_'
                f'{txrx_pol}_{dt_utc_cur}_{dt_utc_first}_{dt_utc_last}.csv'
                )
            file_csv = os.path.join(args.out_path, name_csv)
            logger.info(
                'Dump EL Rising-edge product in "CSV" format to file ->\n'
                f' {file_csv}'
                )
            # dump product into CSV file
            with open(file_csv, 'wt') as fid_csv:
                fid_csv.write('UTC Time,Band,Null Number,Range (m),Elevation'
                              ' (deg),Quality Factor\n')
                # report edge-only product (null=0) w/ quality checking
                # afterwards
                for nn, roll in enumerate(el_ofs):
                    # Simply report the last (rising edge) EL angles and
                    # slant ranges.
                    el_true_deg = np.rad2deg(el_fl[nn, 1])

                    # Quality factor is defined by boolean product of valid
                    # range bin mask and the convergence in rising edge cost
                    # function. Thus the value will be limited to either
                    # 0 (bad) or 1 (good).
                    quality_factor = int(msk[nn] & cvg[nn])

                    # Null number for edge product is zero!
                    fid_csv.write(
                        '{:s},{:1s},{:d},{:.3f},{:.3f},{:.6f}\n'.format(
                            az_dtm[nn].isoformat_usec(), sar_band_char, 0,
                            sr_fl[nn, 1], el_true_deg, quality_factor
                            )
                        )
                    # report possible invalid items/Rows
                    # add two for header line + 0-based index to "nn"
                    if not msk[nn]:
                        logger.warning(
                            f'Row # {nn + 2} may have invalid slant range '
                            'due to TX gap overlap!')
                    if not cvg[nn]:
                        logger.warning(
                            f'Row # {nn + 2} may have invalid roll/EL offset'
                            ' due to failed convergence in rising edge cost '
                            'function!'
                            )

    # total elapsed time
    logger.info(f'Elapsed time -> {time.time() - tic:.1f} (sec)')


def dt2str(dt: 'isce3.core.DateTime', fmt: str = '%Y%m%dT%H%M%S') -> str:
    """isce3 DateTime to a desired string format."""
    return datetime.fromisoformat(dt.isoformat().split('.')[0]).strftime(fmt)


if __name__ == '__main__':
    """Main driver"""
    gen_el_rising_edge_product(cmd_line_parser())
