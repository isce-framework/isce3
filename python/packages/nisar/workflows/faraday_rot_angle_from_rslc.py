#!/usr/bin/env python3
"""
A workflow to extract polarimetric channel imbalance from RSLC products
"""
import os
import numpy as np
import time
import argparse as argp
import json
from datetime import datetime, timezone
from collections import defaultdict

from nisar.products.readers.SLC import SLC
from nisar.log import set_logger
from nisar.workflows.gen_el_null_range_product import dt2str
from nisar.cal import (FaradayRotEstBickelBates,
                       FaradayRotEstFreemanSecond,
                       faraday_rot_angle_from_cr,
                       est_cr_az_mid_swath_from_slc)
from nisar.workflows.pol_channel_imbalance_from_rslc import (
    cr_llh_from_csv, JsonNumpyEncoder
)
from isce3.core import DateTime, TimeDelta


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
        description=(
            'Estimate Faraday rotation angle from extended scene (and, '
            'optionally, corner reflectors) of polarimetrically-balanced'
            ' linear quad-pol RSLC product.'
        ),
        formatter_class=argp.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@"
    )
    prs.add_argument('slc', type=str,
                     help='Filename of linear quad-pol RSLC HDF5 product')
    prs.add_argument('--csv-cr', type=str, dest='csv_cr',
                     help='Filename of UAVSAR/NISAR compatible CSV file for '
                     'corner reflectors (CR) within RSLC product.')
    prs.add_argument('-f', '--freq', type=str, choices=['A', 'B'], default='A',
                     dest='freq_band', help='Frequency band such as "A"')
    prs.add_argument('-m', '--method', type=str, choices=['BB', 'FS', 'SLOPE'],
                     default='BB', dest='method',
                     help=('Method for Faraday rotation (FR) angle estimator. '
                           '"BB" is Bickel-Bates estimator. '
                           '"FS" is Freeman-second estimator.'
                           '"SLOPE" is frequency domain  version of'
                           'Bickel-Bates estimator based on slope')
                     )
    prs.add_argument('-o', '--out-dir', type=str, dest='out_dir', default='.',
                     help='Output directory used for tmp dir, dumping'
                     ' JSON files FaradayRotAngleSLC*.json '
                     '(and FaradayRotAngleCR*.json) containing Faraday '
                     'rotation product from extended scene '
                     '(and corner reflector if any), and for PNG files'
                     ' if `--plot`.')
    prs.add_argument('--sr-lim', nargs=2, type=float, default=(None, None),
                     dest='sr_lim', help='Slant range limits [first, last] '
                     'of extended scene in (m). Default is over all slant '
                     'ranges. This is not used for CR.')
    prs.add_argument('--azt-lim', nargs=2, default=(None, None),
                     dest='azt_lim', help='Azimuth datetime or time limits '
                     '[first, last] of extended scene in (UTC ISO8601 or sec).'
                     ' Default is over entire azimuth duration of `slc`. '
                     'This is not used for CR.')
    prs.add_argument('--sr-spacing', type=float, default=3000.0,
                     dest='sr_spacing', help='Slant range spacing (in meters)'
                     ' per range block of `slc` within the limits of '
                     '`sr_lim`. This is not used for CR.')
    prs.add_argument('--azt-spacing', type=float, default=5.0,
                     dest='azt_spacing', help='Azimuth time spacing '
                     '(in seconds) per azimuth block of `slc` within the '
                     'limits of `azt-lim`. This is not used for CR.')
    prs.add_argument('--average', action='store_true', dest='average',
                     help='If set, it will report a weighted averaged Faraday'
                     'rotation angle over all blocks of an extended scene. '
                     'Otherwise, a 2-D array as a function of range and '
                     'azimuth. This does not apply to estimates from CR. '
                     'Note that the normalized magnitudes of FR estimator '
                     'over all blocks will be used as weights in weighted'
                     ' average.')
    prs.add_argument('--plot', action='store_true', dest='plot',
                     help='Generate PNG plots of Faraday rotation angles.')
    return prs.parse_args()


def faraday_rot_angle_from_rslc(args):
    """Faraday rotation angle from linear quad-pol RSLC product.

    It generates one JSON file for FR angle estimates from extended scene and
    one JSON file for FR angle estimates from CR(s) if CSV file for CR is
    provided.

    Parameters
    ----------
    args : argparse.Namespace
        All input arguments parsed from a command line or an ASCII file.

    References
    ----------
    .. [BICKEL1965] S. B. Bickel and R. H. T. Bates, 'Effects of magneto-ionic
        propagation on the polarization scattering matrix,' Proc. IEEE, vol 53,
        pp. 1089-1091, August 1965.
    .. [Freeman2004] A. Freeman, 'Calibration of linearly polarized
        polarimetric SAR data subject to Faraday rotation,' IEEE Trans.
        Geosci. Remote Sens., Vol 42, pp.1617-1624, August, 2004.
    .. [PAPATHAN2017] K. P. Papathanassiou and J. S. Kim, 'Polarimetric system
        calibration in the presence of Faraday rotation,' Proc. IGARSS IEEE,
        pp. 2430-2433, 2017.

    """
    tic = time.time()
    # set logging
    logger = set_logger("FaradayRotAngleFromRSlc")
    # report some settings
    logger.info(f'Frequency band for RSLC -> "{args.freq_band}"')
    logger.info(
        'Whether to average Faraday rotation angles over all blocks -> '
        f'{args.average}'
    )
    # parse rslc products
    slc = SLC(hdf5file=args.slc)

    # get azimuth time limits in right format, either float(sec) or
    # isce3.core.Datetime(utc)
    azt_lim = list(args.azt_lim)
    if args.azt_lim[0] is not None:
        try:
            azt_lim[0] = float(args.azt_lim[0])
        except ValueError:
            azt_lim[0] = DateTime(args.azt_lim[0])
    if args.azt_lim[1] is not None:
        try:
            azt_lim[1] = float(args.azt_lim[1])
        except ValueError:
            azt_lim[1] = DateTime(args.azt_lim[1])

    # estimate faraday rotation angle from extended scene of SLC
    if args.method == 'BB':
        with FaradayRotEstBickelBates(
                slc, dir_tmp=args.out_dir, plot=args.plot, logger=logger
        ) as fra:

            fra_prod_ext = fra.estimate(
                azt_blk_size=args.azt_spacing, sr_blk_size=args.sr_spacing,
                azt_lim=args.azt_lim, sr_lim=args.sr_lim)

    elif args.method == 'SLOPE':
        with FaradayRotEstBickelBates(
                slc, use_slope_freq=True, dir_tmp=args.out_dir,
                ovsf=1.25, num_cpu_fft=-1, plot=args.plot, logger=logger
        ) as fra:

            fra_prod_ext = fra.estimate(
                azt_blk_size=args.azt_spacing, sr_blk_size=args.sr_spacing,
                azt_lim=args.azt_lim, sr_lim=args.sr_lim)

    elif args.method == 'FS':
        with FaradayRotEstFreemanSecond(
                slc, dir_tmp=args.out_dir, plot=args.plot, logger=logger
        ) as fra:

            fra_prod_ext = fra.estimate(
                azt_blk_size=args.azt_spacing, sr_blk_size=args.sr_spacing,
                azt_lim=args.azt_lim, sr_lim=args.sr_lim)

    else:
        raise NotImplementedError(
            'Supported methods are "BB", "FS", and "SLOPE"!')

    # dump the products for extended scene
    # get the first and last utc azimuth time w/o fractional seconds
    # in "%Y%m%dT%H%M%S" format to be used as part of JSON product filename.
    dt_utc_first = dt2str(fra_prod_ext.az_datetime[0])
    dt_utc_last = dt2str(fra_prod_ext.az_datetime[-1])
    # get current time w/o fractional seconds in "%Y%m%dT%H%M%S" format
    # used as part of JSON product filename
    dt_utc_cur = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    # form filename of the JSON product for extended scene
    name_json_ext = (
        f'FaradayRotAngleSLC_{dt_utc_cur}_{dt_utc_first}_{dt_utc_last}.json'
    )
    # pack and dump the final product (all or average) to a JSON file
    fra_slc_ext = dict()
    if args.average:
        # weighted average over all blocks
        # Use the magnitudes as weights with proper normalization
        fra_slc_ext['faraday_ang_rad'] = np.angle(np.mean(
            fra_prod_ext.magnitude * np.exp(1j * fra_prod_ext.faraday_ang)))
        # store averaged magnitudes as a final magnitude
        fra_slc_ext['magnitude'] = fra_prod_ext.magnitude.mean()
        fra_slc_ext['slant_range'] = fra_prod_ext.slant_range.mean()
        # get mid azimuth DateTime
        azt_rel_mid = np.mean(
            [(dt - fra_prod_ext.az_datetime[0]).total_seconds()
             for dt in fra_prod_ext.az_datetime
             ]
        )
        fra_slc_ext['az_datetime'] = (fra_prod_ext.az_datetime[0] +
                                      TimeDelta(azt_rel_mid)).isoformat()
        if fra_prod_ext.slope_freq is not None:
            fra_slc_ext['slope_freq_rad_per_hz'] = np.mean(
                fra_prod_ext.slope_freq)

    else:  # values per range and azimuth blocks
        fra_slc_ext['faraday_ang_rad'] = fra_prod_ext.faraday_ang
        fra_slc_ext['magnitude'] = fra_prod_ext.magnitude
        fra_slc_ext['slant_range'] = fra_prod_ext.slant_range
        # get isoformat string version of DateTimes
        fra_slc_ext['az_datetime'] = [
            dt.isoformat() for dt in fra_prod_ext.az_datetime
        ]
        if fra_prod_ext.slope_freq is not None:
            fra_slc_ext['slope_freq_rad_per_hz'] = fra_prod_ext.slope_freq

    # dump final outcome into json file
    with open(os.path.join(args.out_dir, name_json_ext), 'w') as fidw:
        json.dump(fra_slc_ext, fidw, indent=4, cls=JsonNumpyEncoder)

    # parse csv file for CR(s) info if any CSV file is provided
    if args.csv_cr is not None:
        logger.info('FR angles are also estimated from CRs!')
        # get start datetime of the RSLC product
        rdr_grid = slc.getRadarGrid(args.freq_band)
        epoch_start = rdr_grid.sensing_datetime(0)
        # Get CRs that are facing in the right direction based on AZ angle
        # of a CR and optimum heading at roughly mid swath/footprint.
        cr_az_ang = est_cr_az_mid_swath_from_slc(slc)

        cr_llh = cr_llh_from_csv(
            args.csv_cr, epoch=epoch_start, az_heading=cr_az_ang)

        fra_prod_crs = faraday_rot_angle_from_cr(
            slc, cr_llh, freq_band=args.freq_band
        )
        logger.info(
            f'Number of detected CRs within SLC -> {len(fra_prod_crs)}'
        )
        # pack and dump the final product to a JSON file for all CRs
        fra_slc_crs = defaultdict(list)
        for fra_cr in fra_prod_crs:
            fra_slc_crs['faraday_ang_rad'].append(fra_cr.faraday_ang)
            fra_slc_crs['magnitude'].append(fra_cr.magnitude)
            fra_slc_crs['slant_range'].append(fra_cr.slant_range)
            fra_slc_crs['az_datetime'].append(fra_cr.az_datetime.isoformat())
            fra_slc_crs['longitude_rad'].append(fra_cr.llh.longitude)
            fra_slc_crs['latitude_rad'].append(fra_cr.llh.latitude)
            fra_slc_crs['height'].append(fra_cr.llh.height)
            fra_slc_crs['el_ant_rad'].append(fra_cr.el_ant)
            fra_slc_crs['az_ant_rad'].append(fra_cr.az_ant)
            if fra_cr.slope_freq is not None:
                fra_slc_crs['slope_freq_rad_per_hz'].append(fra_cr.slope_freq)

        # get first and last azimuth time of entire SLC given azimuth time
        # of RSLC is not limited for CRs. Convert them into "%Y%m%dT%H%M%S"
        # format
        dt_first = dt2str(epoch_start)
        dt_last = dt2str(rdr_grid.sensing_datetime(rdr_grid.length - 1))
        # form the name for JSON product and dump the results
        name_json_cr = (
            f'FaradayRotAngleCR_{dt_utc_cur}_{dt_first}_{dt_last}.json'
        )
        with open(os.path.join(args.out_dir, name_json_cr), 'w') as fidw:
            json.dump(fra_slc_crs, fidw, indent=4)

    # total elapsed time
    logger.info(f'Elapsed time -> {time.time() - tic:.1f} (sec)')


if __name__ == "__main__":
    """Main driver"""
    faraday_rot_angle_from_rslc(cmd_line_parser())
