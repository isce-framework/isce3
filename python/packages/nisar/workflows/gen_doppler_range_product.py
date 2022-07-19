#!/usr/bin/env python3
"""
Generate Doppler Centroid product from L0B data
"""
import os
import time
import argparse as argp
import numpy as np
from datetime import datetime

from nisar.workflows import doppler_lut_from_raw
from nisar.workflows.doppler_lut_from_raw import set_logger
from nisar.products.readers.Raw import Raw
from isce3.core import TimeDelta, Linspace
from nisar.products.readers.antenna import AntennaParser
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
        description='Estimate Doppler centroid from L0B raw echo and creates '
        'a 2-D Doppler LUT dumped into a CSV file',
        fromfile_prefix_chars="@",
        formatter_class=argp.ArgumentDefaultsHelpFormatter
    )
    prs.add_argument('filename_l0b', type=str,
                     help='Filename of HDF5 L0B product')
    prs.add_argument('--antenna_file', type=str, dest='antenna_file',
                     help='Filename of HDF5 Antenna product used to extract '
                     'averaged azimuth angle for EL cuts of TX + RX pol of '
                     'first beam. It also uses for Doppler ambiguity '
                     'calculation. If not provided, the azimuth angle is '
                     'assumed to be zero!')
    prs.add_argument('-f', '--freq', type=str, choices=['A', 'B'], default='A',
                     dest='freq_band', help='Frequency band such as "A".')
    prs.add_argument('-p', '--pol', type=str, dest='txrx_pol',
                     choices=["HH", "VV", "HV", "VH"],
                     help='TxRx Polarization such as "HH". Default is the '
                     'first pol in the specified frequency band')
    prs.add_argument('-r', '--rgb', type=int, dest='num_rgb_avg', default=16,
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

    # operation mode, whether DBF (single or a composite channel) or
    # 'DM2' (multi-channel)
    # currently, the nunderlying module simply support DBF or single channel
    op_mode = 'DBF'

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
        ant = None
    else:
        ant = AntennaParser(args.antenna_file)

    # get keyword args for function "doppler_lut_from_raw"
    kwargs = {key: val for key, val in vars(args).items() if
              'file' not in key}

    # generate Doppler LUT2d from Raw L0B
    dop_lut, ref_utc, mask_rgb, corr_coef, txrx_pol, centerfreq, _ = \
        doppler_lut_from_raw(raw_obj, orbit=orbit, attitude=attitude,
                             ant=ant, logger=logger,  **kwargs)

    # check out antenna object to extract azimuth angle for EL cuts used for
    # Doppler CSV product
    if ant is None:
        az_ang_deg = 0.0
        logger.warning(
            'No antenna file! Azimuth angle for Doppler product is '
            'assumed to be zero!'
        )
    else:
        logger.info(
            'Extracting the azimuth angle of EL cuts from antenna file.')

        ant_rx = ant.el_cut(pol=txrx_pol[1])
        az_ang = ant_rx.cut_angle
        # if TX pol is different from RX pol then take average of both
        if txrx_pol[0] != txrx_pol[1]:
            # To cover TX L/R circular pol, the following condition is needed
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

    # form Linspace object for uniformly-spaced azimuth time and slant range
    azt_lsp = Linspace(dop_lut.y_start, dop_lut.y_spacing, dop_lut.length)
    sr_lsp = Linspace(dop_lut.x_start, dop_lut.x_spacing, dop_lut.width)

    # get the first and last utc azimuth time w/o fractional seconds
    # in "%Y%m%dT%H%M%S" format to be used as part of CSV product filename.
    dt_utc_start = sec2str(ref_utc, azt_lsp.first)
    dt_utc_stop = sec2str(ref_utc, azt_lsp.last)
    # get current time w/o fractional seconds in "%Y%m%dT%H%M%S" format
    # used as part of CSV product filename
    dt_utc_cur = datetime.now().strftime('%Y%m%dT%H%M%S')

    # naming convention of CSV file and product spec is defined in Doc:
    # See reference [1]
    name_csv = (f'{PREFIX_NAME_CSV}_{sar_band_char}_{op_mode}_DOPP_'
                f'{dt_utc_cur}_{dt_utc_start}_{dt_utc_stop}.csv')
    file_csv = os.path.join(args.out_path, name_csv)
    logger.info(f'Dump Doppler product in "CSV" format to file -> {file_csv}')

    with open(file_csv, 'wt') as fid_csv:
        fid_csv.write(
            'UTC Time,Frequency (Hz),Doppler (Hz),Range (m),Azimuth (deg),'
            'Correlation\n'
        )
        # loop over azimuth time and slant ranges
        for i_row, azt in enumerate(azt_lsp):
            tm_utc_str = sec2isofmt(ref_utc, azt)

            for i_col, sr in enumerate(sr_lsp):
                fid_csv.write(
                    '{:s},{:.1f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
                        tm_utc_str, centerfreq, dop_lut.data[i_row, i_col],
                        sr, az_ang_deg,
                        mask_rgb[i_row, i_col] * corr_coef[i_row, i_col])
                )

    # total elapsed time
    logger.info(f'Elapsed time -> {time.time() - tic:.1f} (sec)')


def sec2isofmt(ref_utc: 'isce3.core.DateTime', seconds: float) -> str:
    """seconds to isoformat string"""
    return (ref_utc + TimeDelta(seconds)).isoformat()


def sec2str(ref_utc: 'isce3.core.DateTime', seconds: float) -> str:
    """seconds to string format '%Y%m%dT%H%M%S'"""
    fmt = '%Y%m%dT%H%M%S'
    dt_iso = sec2isofmt(ref_utc, seconds)
    return datetime.fromisoformat(dt_iso.split('.')[0]).strftime(fmt)


if __name__ == "__main__":
    """Main driver"""
    gen_doppler_range_product(cmd_line_parser())
