#!/usr/bin/env python3
"""
Generate El Null Range product from L0B (DM2) + Antenna HDF5 + [DEM raster]
data which will be used for pointing analyses by D&C team
"""
import os
import time
import argparse as argp
from datetime import datetime

from nisar.products.readers.Raw import Raw
from nisar.products.readers.antenna import AntennaParser
from isce3.geometry import DEMInterpolator
from isce3.io import Raster
from nisar.workflows import el_null_range_from_raw_ant
from nisar.workflows.doppler_lut_from_raw import set_logger
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
    prs.add_argument('-f', '--freq', type=str, choices=['A', 'B'], default='A',
                     dest='freq_band', help='Frequency band such as "A"')
    prs.add_argument('-p', '--pol', type=str, dest='txrx_pol',
                     choices=['HH', 'VV', 'HV', 'VH'],
                     help=('TxRx Polarization such as "HH".Default is the '
                           'first pol in the band "A"')
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

    # get keyword args for function "el_null_range_from_raw_ant"
    kwargs = {key: val for key, val in vars(args).items() if
              key in ['az_block_dur', 'freq_band', 'txrx_pol',
                      'apply_caltone']}
    (null_num, sr_echo, el_ant, pow_ratio, az_datetime, null_flag, mask_valid,
     txrx_pol, wavelength) = el_null_range_from_raw_ant(
         raw_obj, ant_obj, dem_interp_obj, logger=logger, orbit=orbit,
         attitude=attitude, **kwargs)

    # get the first and last utc azimuth time w/o fractional seconds
    # in "%Y%m%dT%H%M%S" format to be used as part of CSV product filename.
    dt_utc_first = dt2str(az_datetime[0])
    dt_utc_last = dt2str(az_datetime[-1])
    # get current time w/o fractional seconds in "%Y%m%dT%H%M%S" format
    # used as part of CSV product filename
    dt_utc_cur = datetime.now().strftime('%Y%m%dT%H%M%S')

    # naming convention of CSV file and product spec is defined in Doc:
    # See reference [1]
    name_csv = (
        f'{prefix_name_csv}_{sar_band_char}{args.freq_band}_{op_mode}_NULL_'
        f'{txrx_pol}_{dt_utc_cur}_{dt_utc_first}_{dt_utc_last}.csv'
        )
    file_csv = os.path.join(args.out_path, name_csv)
    logger.info(
        f'Dump EL Null-Range product in "CSV" format to file ->\n {file_csv}')
    # dump product into CSV file
    with open(file_csv, 'wt') as fid_csv:
        fid_csv.write('UTC Time,Band,Null Number,Range (m),Elevation (deg),'
                      'Quality Factor\n')
        # report null-only product (null # > 0) w/ quality checking afterwards
        for nn in range(null_num.size):
            fid_csv.write('{:s},{:1s},{:d},{:.3f},{:.3f},{:.6f}\n'.format(
                az_datetime[nn].isoformat(), sar_band_char, null_num[nn],
                sr_echo[nn], el_ant[nn], 1 - pow_ratio[nn]))
            # report possible invalid items/Rows
            # add three for header line + null_zero + 0-based index to "nn"
            if not mask_valid[nn]:
                logger.warning(f'Row # {nn + 2} may have invalid slant range '
                               'due to TX gap overlap!')
            if not null_flag[nn]:
                logger.warning(
                    f'Row # {nn + 3} may have invalid slant range due to '
                    'failed convergence in null location estimation!')

    # total elapsed time
    logger.info(f'Elapsed time -> {time.time() - tic:.1f} (sec)')


def dt2str(dt: 'isce3.core.DateTime', fmt: str = '%Y%m%dT%H%M%S') -> str:
    """isce3 DateTime to a desired string format."""
    return datetime.fromisoformat(dt.isoformat().split('.')[0]).strftime(fmt)


if __name__ == "__main__":
    """Main driver"""
    gen_el_null_range_product(cmd_line_parser())
