#!/usr/bin/env python3
"""
Generate El Null Range product from L0B (DM2) + Antenna HDF5 + [DEM raster]
data which will be used for pointing analyses by D&C team
"""
from __future__ import annotations
import os
import time
import argparse as argp
from datetime import datetime, timezone

from nisar.products.readers.Raw import Raw
from nisar.products.readers.antenna import AntennaParser
from isce3.geometry import DEMInterpolator
from isce3.io import Raster
from nisar.pointing import el_null_range_from_raw_ant
from nisar.log import set_logger
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.products.readers.attitude import load_attitude_from_xml


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
        description=('Generate EL Null-Range product from L0B (DM2) + Antenna '
                     'HDF5 + [DEM raster] data'),
        formatter_class=argp.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@"
    )
    prs.add_argument('--l0b', type=str, required=True, dest='l0b_file',
                     help='Filename of HDF5 L0B product, Diagnostic Mode # 2')
    prs.add_argument('--ant', type=str, required=True, dest='antenna_file',
                     help='Filename of HDF5 Antenna product')
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
                     help='Output directory to dump EL null product. The '
                     'product is CSV file whose name conforms to JPL D-104976.'
                     )
    prs.add_argument('--caltone', action='store_true', dest='apply_caltone',
                     help=('Balance out RX channels by applying caltone '
                           'coefficients extracted from L0B.')
                     )
    prs.add_argument('-r', '--ref_height', type=float, dest='ref_height',
                     default=0.0,
                     help=('Reference height in (m) w.r.t WGS84 ellipsoid. It '
                           'will be simply used if "dem_file" is not provided')
                     )
    prs.add_argument('--plot', action='store_true', dest='plot',
                     help='Plot null power patterns (antenna v.s. echo) and '
                     'save them in *.png files at the specified output path')
    prs.add_argument('--deg', type=int, dest='polyfit_deg',
                     default=6, help='Degree of the polyfit used for'
                     ' smoothing and location estimation of echo null.')

    return prs.parse_args()


def gen_el_null_range_product(args):
    """Generate EL Null-Range Product.

    It generates Null locations in elevation (EL) direction as a function
    slant range at various azimuth/pulse times and dump them into a CSV file.

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
    # 'DM2' (multi-channel). Null only possible via "DM2"!
    op_mode = 'DM2'

    tic = time.time()
    # set logging
    logger = set_logger("ELNullRangeProduct")

    # build Raw object
    raw_obj = Raw(hdf5file=args.l0b_file)
    # get the SAR band char
    sar_band_char = raw_obj.sarBand
    logger.info(f'SAR band char -> {sar_band_char}')

    # build antenna object from antenna file
    ant_obj = AntennaParser(args.antenna_file)

    # build dem interp object from DEM raster or ref height
    if args.dem_file is None:  # set to a fixed height
        dem_interp_obj = DEMInterpolator(args.ref_height)
    else:  # build from DEM Raster file
        dem_interp_obj = DEMInterpolator(Raster(args.dem_file))

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

    # get common keyword args for function "el_null_range_from_raw_ant"
    kwargs = {key: val for key, val in vars(args).items() if
              key in ['az_block_dur', 'apply_caltone', 'plot',
                      'out_path', 'polyfit_deg']}

    # logic for frequency band and TxRx polarization choices.
    # form a new dict "frq_pol" with key=freq_band and value=[txrx_pol]
    frq_pol = copol_or_desired_product_from_raw(
        raw_obj, args.freq_band, args.txrx_pol)
    logger.info(f'List of selected frequency bands and TxRx Pols -> {frq_pol}')

    # loop over all desired frequency bands and their respective desired
    # polarizations
    for freq_band in frq_pol:
        for txrx_pol in frq_pol[freq_band]:

            (null_num, sr_echo, el_ant, pow_ratio, az_datetime, null_flag,
             mask_valid, _, wavelength) = el_null_range_from_raw_ant(
                 raw_obj, ant_obj, dem_interp=dem_interp_obj, logger=logger,
                 orbit=orbit, attitude=attitude, freq_band=freq_band,
                 txrx_pol=txrx_pol, **kwargs
            )
            # get the first and last utc azimuth time w/o fractional seconds
            # in "%Y%m%dT%H%M%S" format to be used as part of CSV product
            # filename.
            dt_utc_first = dt2str(az_datetime[0])
            dt_utc_last = dt2str(az_datetime[-1])
            # get current time w/o fractional seconds in "%Y%m%dT%H%M%S" format
            # used as part of CSV product filename
            dt_utc_cur = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')

            # naming convention of CSV file and product spec is defined in Doc:
            # See reference [1]
            name_csv = (
                f'{prefix_name_csv}_{sar_band_char}{freq_band}_{op_mode}_NULL_'
                f'{txrx_pol}_{dt_utc_cur}_{dt_utc_first}_{dt_utc_last}.csv'
            )
            file_csv = os.path.join(args.out_path, name_csv)
            logger.info(
                'Dump EL Null-Range product in "CSV" format to file ->\n '
                f'{file_csv}')
            # dump product into CSV file
            with open(file_csv, 'wt') as fid_csv:
                fid_csv.write('UTC Time,Band,Null Number,Range (m),Elevation'
                              ' (deg),Quality Factor\n')
                # report null-only product (null # > 0) w/ quality checking
                # afterwards
                for nn in range(null_num.size):
                    fid_csv.write(
                        '{:s},{:1s},{:d},{:.3f},{:.3f},{:.3f}\n'.format(
                            az_datetime[nn].isoformat_usec(), sar_band_char,
                            null_num[nn], sr_echo[nn], el_ant[nn],
                            1 - pow_ratio[nn])
                    )
                    # report possible invalid items/Rows
                    # add three for header line + null_zero + 0-based index to
                    # "nn"
                    if not mask_valid[nn]:
                        logger.warning(
                            f'Row # {nn + 2} may have invalid slant range '
                            'due to TX gap overlap!')
                    if not null_flag[nn]:
                        logger.warning(
                            f'Row # {nn + 3} may have invalid slant range due'
                            ' to failed convergence in null location '
                            'estimation!'
                        )

    # total elapsed time
    logger.info(f'Elapsed time -> {time.time() - tic:.1f} (sec)')


def dt2str(dt: 'isce3.core.DateTime', fmt: str = '%Y%m%dT%H%M%S') -> str:
    """isce3 DateTime to a desired string format."""
    return datetime.fromisoformat(dt.isoformat().split('.')[0]).strftime(fmt)


def copol_or_desired_product_from_raw(
        raw: Raw, freq_band: str | None = None, txrx_pol: str | None = None
        ) -> dict:
    """
    Fetch either all co-pol products from Raw (default) or desired ones which
    fulfill either parameter (`freq_band` and/or `txrx_pol`) if provided. The
    output is in the form of a dict with key=freq_band and value=[TxRx_pol].
    """
    # get list of frequency bands
    if freq_band is None:
        freqs = list(raw.polarizations)
    else:
        # go with desired frequency band
        if freq_band not in raw.polarizations:
            raise ValueError('Wrong frequency band! The available bands -> '
                             f'{list(raw.polarizations)}')
        freqs = [freq_band]

    frq_pol = dict()
    for frq in freqs:
        pols = raw.polarizations[frq]
        if txrx_pol is None:
            # get all co-pols if pol is not provided
            co_pols = [pol for pol in pols if (pol[0] == pol[1] or
                                               pol[0] in ['L', 'R'])]
            if co_pols:
                frq_pol[frq] = co_pols
        else:
            # get all pols over all bands that match the desired one
            if txrx_pol in pols:
                frq_pol[frq] = [txrx_pol]

    # check if the dict empty (it simply occurs if txrx_pol is provided!)
    if not frq_pol:
        raise ValueError(f'Wrong TxRx Pol over frequency bands {freqs}!')

    return frq_pol


if __name__ == "__main__":
    """Main driver"""
    gen_el_null_range_product(cmd_line_parser())
