#!/usr/bin/env python3
"""
A workflow to extract polarimetric channel imbalance from RSLC products
"""
import os
import numpy as np
import time
import argparse as argp
import json
from datetime import datetime
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from nisar.products.readers.SLC import SLC
from nisar.workflows.doppler_lut_from_raw import set_logger
from nisar.workflows.gen_el_null_range_product import dt2str
from nisar.cal import PolChannelImbalanceSlc
from isce3.antenna import CrossTalk
from isce3.geometry import DEMInterpolator
from isce3.io import Raster
from isce3.core import DateTime


class JsonNumpyEncoder(json.JSONEncoder):
    """
    A thin wrapper around JSONEncoder w/ augmented default method to support
    various numpy array and complex data types
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, (complex, np.complexfloating)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        return super().default(obj)


def cr_llh_from_csv(filename, skip_first_row=True):
    """Parse LLH of all corner reflectors (CRs) from a CSV file.

    It supports both UAVSAR-formatted CSV or the truncated version with
    simply four columns:
    "CR ID, latitude(deg), longitude (deg), and height (m)".

    Parameters
    ----------
    filename : str
        Filename of csv file.
    skip_first_row : bool, default=True
        If True the first row is assumed to be a comment line.

    Returns
    -------
    np.ndarray(np.ndarray(float))
        2-D array with shape (N, 3) where `N` is the number of CRs.
        The three values represents longitude, latitude, height in
        (rad, rad, m).

    """
    cr_llh = np.loadtxt(
        filename, delimiter=',', skiprows=int(skip_first_row),
        usecols=range(1, 4), ndmin=2)
    cr_llh[:, -2::-1] = np.deg2rad(cr_llh[:, :2])
    return cr_llh


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
        description='Extract TX/RX Polarimetric Channel Imbalances from RSLCs',
        formatter_class=argp.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@"
    )
    prs.add_argument('--slc-ext', type=str, required=True, dest='slc_ext',
                     help='Filename of quad-pol RSLC HDF5 product over '
                     'extended homogenous scene')
    prs.add_argument('--slc-cr', type=str, required=True, dest='slc_cr',
                     help='Filename of quad-pol RSLC HDF5 product over corner '
                     'reflector.')
    prs.add_argument('--csv-cr', type=str, required=True, dest='csv_cr',
                     help='Filename of UAVSAR-compatible CSV file for corner '
                     'reflectors.')
    prs.add_argument('-f', '--freq', type=str, choices=['A', 'B'], default='A',
                     dest='freq_band', help='Frequency band such as "A"')
    prs.add_argument('-d', '--dem-file', type=str, dest='dem_file',
                     help='DEM raster file in (GDAL-compatible format such as '
                     'GeoTIFF) containing heights w.r.t. WGS-84 ellipsoid. '
                     'Default is no DEM!')
    prs.add_argument('-r', '--ref-height', type=float, dest='ref_height',
                     default=0.0,
                     help='Reference height in (m) w.r.t WGS84 ellipsoid. It '
                     'will be simply used if "dem_file" is not provided')
    prs.add_argument('-o', '--out-dir', type=str, dest='out_dir', default='.',
                     help='Output directory used for tmp dir/files, dumping'
                     ' JSON file containing channel imbalances for both '
                     'TX and RX, and for PNG files if `--plot`.')
    prs.add_argument('--ignore-faraday', action='store_true',
                     dest='ignore_faraday', help='If set, it is assumed that'
                     ' Faraday rotation is zero!')
    prs.add_argument('--tx-xtalk-amp', nargs=2, type=float, default=[0.0, 0.0],
                     dest='tx_xtalk_amp', help='Cross-talk amplitudes for TX '
                     '[H, V] pols in linear scale')
    prs.add_argument('--tx-xtalk-phs', nargs=2, type=float, default=[0.0, 0.0],
                     dest='tx_xtalk_phs', help='Cross-talk phases for TX '
                     '[H, V] pols in radians')
    prs.add_argument('--rx-xtalk-amp', nargs=2, type=float, default=[0.0, 0.0],
                     dest='rx_xtalk_amp', help='Cross-talk amplitudes for RX '
                     '[H, V] pols in linear scale')
    prs.add_argument('--rx-xtalk-phs', nargs=2, type=float, default=[0.0, 0.0],
                     dest='rx_xtalk_phs', help='Cross-talk phases for RX '
                     '[H, V] pols in radians')
    prs.add_argument('--sr-lim', nargs=2, type=float, default=(None, None),
                     dest='sr_lim', help='Slant range limits [first, last] '
                     'of `slc_ext` in (m). Default is over all slant ranges.')
    prs.add_argument('--azt-lim', nargs=2, default=(None, None),
                     dest='azt_lim', help='Azimuth datetime or time limits '
                     '[first, last] of `slc_ext` in (UTC ISO8601 or sec). '
                     'Default is over entire azimuth duration.')
    prs.add_argument('--sr-spacing', type=float, default=3000.0,
                     dest='sr_spacing', help='Slant range spacing per range'
                     ' blocks of `slc_ext` within the limits of `sr_lim`')
    prs.add_argument('--azt-spacing', type=float, default=5.0,
                     dest='azt_spacing', help='Azimuth time spacing per '
                     'azimuth blocks of `slc_ext` within the limits of'
                     ' `sr_lim`')
    prs.add_argument('--mean-el', action='store_true', dest='mean_el',
                     help='If set, it will report a mean complex value for '
                     'TX/RX over all EL angles. Otherwise, a vector of values'
                     ' will be reported at various EL angles if any.')
    prs.add_argument('--plot', action='store_true', dest='plot',
                     help='Generate plots of TX/RX channel imbalances in PNG.')
    return prs.parse_args()


def pol_channel_imbalance_from_rslc(args):
    """Polarimetric channel imbalance from RSLC products.

    It generates a JSON file with complex polarimetric channel imbalance as
    a function of antenna elevation (EL) angle for both TX and RX.

    Parameters
    ----------
    args : argparse.Namespace
        All input arguments parsed from a command line or an ASCII file.

    """
    tic = time.time()
    # set logging
    logger = set_logger("PolChannelImbalanceFromRSlc")
    # report some settings
    logger.info(f'Frequency band for all RSLCs -> "{args.freq_band}"')
    logger.info(f'Assume no faraday rotation -> {args.ignore_faraday}')
    logger.info(
        f'Mean channel imbalances over all EL angles -> {args.mean_el}'
    )
    # form a cross talk object
    tx_xtalk = np.asarray(args.tx_xtalk_amp) * np.exp(
        1j * np.asarray(args.tx_xtalk_phs))
    rx_xtalk = np.asarray(args.rx_xtalk_amp) * np.exp(
        1j * np.asarray(args.rx_xtalk_phs))
    xtalk = CrossTalk(*tx_xtalk, *rx_xtalk)

    # build dem interp object from DEM raster or ref height
    if args.dem_file is None:  # set to a fixed height
        dem = DEMInterpolator(args.ref_height)
    else:  # build from DEM Raster file
        dem = DEMInterpolator(Raster(args.dem_file))

    # parse rslc products
    slc_ext = SLC(hdf5file=args.slc_ext)
    if args.slc_ext == args.slc_cr:
        logger.warning(f'The same SLC file "{args.slc_ext}" will be used'
                       ' for homogenous extended scene and CRs!')
        slc_cr = slc_ext
    # separate product/file for CR(s)
    else:
        slc_cr = SLC(hdf5file=args.slc_cr)

    # parse csv file for CR(s) info
    cr_llh = cr_llh_from_csv(args.csv_cr)

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

    # estimate pol imbalances for both TX and RX
    with PolChannelImbalanceSlc(
            slc_ext, slc_cr, cr_llh, dir_tmp=args.out_dir, cross_talk=xtalk,
            freq_band=args.freq_band, ignore_faraday=args.ignore_faraday,
            logger=logger, dem=dem) as pci:

        imb_prod_ant, imb_prod_slc = pci.estimate(
            azt_blk_size=args.azt_spacing, sr_blk_size=args.sr_spacing,
            azt_lim=azt_lim, sr_lim=args.sr_lim
        )
    # get the first and last utc azimuth time w/o fractional seconds
    # in "%Y%m%dT%H%M%S" format to be used as part of JSON product filename.
    dt_utc_first = dt2str(imb_prod_slc.az_datetime[0])
    dt_utc_last = dt2str(imb_prod_slc.az_datetime[-1])
    # get current time w/o fractional seconds in "%Y%m%dT%H%M%S" format
    # used as part of JSON product filename
    dt_utc_cur = datetime.now().strftime('%Y%m%dT%H%M%S')
    # form filename of the JSON product
    name_json = (
        f'PolChannelImbSlc_{dt_utc_cur}_{dt_utc_first}_{dt_utc_last}.json'
    )
    # pack the final output for JSON product
    pol_imb_out = dict()
    if args.mean_el:
        pol_imb_out['tx_pol_ratio'] = imb_prod_ant.tx_pol_ratio.y.mean()
        pol_imb_out['rx_pol_ratio'] = imb_prod_ant.rx_pol_ratio.y.mean()
        pol_imb_out['el_rad'] = imb_prod_ant.tx_pol_ratio.x.mean()
    else:
        pol_imb_out['tx_pol_ratio'] = imb_prod_ant.tx_pol_ratio.y
        pol_imb_out['rx_pol_ratio'] = imb_prod_ant.rx_pol_ratio.y
        pol_imb_out['el_rad'] = imb_prod_ant.tx_pol_ratio.x
    # dump final outcome as json product
    with open(os.path.join(args.out_dir, name_json), 'w') as fidw:
        json.dump(pol_imb_out, fidw, indent=4, cls=JsonNumpyEncoder)

    # check if there is matplotlib package needed for plotting if requested
    if args.plot:
        if plt is None:
            logger.warning('No plots due to missing package "matplotlib"!')
        else:  # plot some output
            # 2-D scatter plot of amp (linear) and phase (deg) of TX/RX imb
            # as a function of azimuth time (sec) and slant range (m)
            azt_sec = np.asarray(
                [dt.seconds_of_day() for dt in imb_prod_slc.az_datetime])
            # mesh grid for az time (sec) and slant range (km)
            sr_mesh, azt_mesh = np.meshgrid(imb_prod_slc.slant_range * 1e-3,
                                            azt_sec)
            # plot TX side
            plot_name = os.path.join(args.out_dir,
                                     'Plot_Tx_Pol_Imbalance_Ratio_Slc.png')
            _plot2d(sr_mesh, azt_mesh, imb_prod_slc.tx_pol_ratio, plot_name,
                    tag='TX')
            # plot RX side
            plot_name = os.path.join(args.out_dir,
                                     'Plot_Rx_Pol_Imbalance_Ratio_Slc.png')
            _plot2d(sr_mesh, azt_mesh, imb_prod_slc.rx_pol_ratio, plot_name,
                    tag='RX')

    # total elapsed time
    logger.info(f'Elapsed time -> {time.time() - tic:.1f} (sec)')


def _plot2d(sr_mesh: np.ndarray, azt_mesh: np.ndarray, pol_ratio: np.ndarray,
            plot_name: str, tag: str, ):
    """2-D scatter plots for pol channel imbalance ratio"""
    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.scatter(sr_mesh, azt_mesh, c=abs(pol_ratio), marker='o', cmap='jet')
    plt.colorbar(label='Amp (linear)')
    plt.xlabel('Slant Ranges (km)')
    plt.ylabel('Azimuth Time (sec)')
    plt.grid(True)
    plt.title(f'{tag} Polarimetric Channel Imbalance Amplitude & Phase')
    plt.subplot(212)
    plt.scatter(sr_mesh, azt_mesh, c=np.angle(pol_ratio, deg=True),
                marker='*', cmap='jet')
    plt.colorbar(label='Phase (deg)')
    plt.xlabel('Slant Ranges (km)')
    plt.ylabel('Azimuth Time (sec)')
    plt.grid(True)
    plt.savefig(plot_name)


if __name__ == "__main__":
    """Main driver"""
    pol_channel_imbalance_from_rslc(cmd_line_parser())