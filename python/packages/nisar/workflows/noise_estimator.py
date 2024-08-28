#!/usr/bin/env python3
"""
Workflow for estimating noise power from L0B raw data
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from nisar.log import set_logger
from nisar.products.readers.Raw import Raw
from nisar.noise import est_noise_power_from_raw
from nisar.noise.noise_estimation_from_raw import PERC_INVALID_NOISE
from nisar.workflows.helpers import JsonNumpyEncoder


def cmd_line_parse():
    parser = argparse.ArgumentParser(
        description=(
            'Estimate noise power in DN**2 units for all frequency bands'
            ' and polarizations from a L0B raw product.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument('l0b_file', type=str, help='L0B path and filename')
    parser.add_argument('-n', '--num-rng-block', type=int,
                        dest='num_rng_block',
                        help=('Number of range blocks. Default is min '
                              'recommended `2*C-1` where `C` is number of RX '
                              'channels.')
                        )
    parser.add_argument('-c', '--cpi', type=int, dest='cpi',
                        help=('Coherent processing interval in number of '
                              'range lines. Only used when algorithm=MEE. '
                              'Min required value is 2! Default is set to '
                              'min 3 while the max number of CPI blocks is '
                              'limited to 8 if possible.')
                        )
    parser.add_argument('-a', '--algorithm', dest='algorithm', type=str,
                        default='MEE', choices=('MVE', 'MEE'),
                        help=('Noise Estimation algorithms: '
                              'MVE (Min Variance Estimator) per sample var, '
                              'MEE (Min Eigenvalue Estimator)')
                        )
    parser.add_argument('-o', '--output', dest='output_path', type=str,
                        default='.',
                        help='Output directory for PNG plots if `--plot`.')
    parser.add_argument('-j', '--json', dest='json_file', type=str,
                        default='noise_power_est_info.json',
                        help='Output JSON path and filename.')
    parser.add_argument('--no-diff', action='store_true',
                        dest='no_diff',
                        help=('If True, it will not differentiate dataset '
                              'w.r.t to the one of the range line for each '
                              'product when algorithm=MVE.')
                        )
    parser.add_argument('--diff-method', type=str, default='single',
                        dest='diff_method',
                        choices=['single', 'mean', 'diff'],
                        help=('Method for differentiating range lines for '
                              'algorithm=MVE if diff is True.')
                        )
    parser.add_argument('--diff-quad', action='store_true',
                        dest='diff_quad',
                        help=('If set, it will differentiate Co/Cx-pol '
                              'datasets with the same RX pol in a joint '
                              'noise est for only QP case.')
                        )
    parser.add_argument('--pct-invalid-rngblk', type=float,
                        dest='pct_invalid_rngblk',
                        default=PERC_INVALID_NOISE,
                        help=('Threshold in percentage of invalid range bins '
                              'within range block above which the block is '
                              'skipped and noise power is set to NaN. '
                              'A value within [0, 100]!')
                        )
    parser.add_argument('--exclude-first-last', action='store_true',
                        dest='exclude_first_last',
                        help=('First and last noise-only range lines will be '
                              'excluded in noise estimation. This can '
                              'help avoid possible outliers in some '
                              'mixed-mode NISAR cases.'
                              )
                        )
    parser.add_argument('--no-median-ev', action='store_true',
                        dest='no_median_ev',
                        help=('If set, no median of eigenvals but '
                              'simply min eigenval as noise power in MEE.')
                        )
    parser.add_argument('--plot', action='store_true', dest='plot',
                        help='Generates PNG plots of noise power at `output`.'
                        )

    return parser.parse_args()


def run_noise_estimator(args: argparse.Namespace):
    """Run CLI noise estimator"""
    tic = time.time()
    # set logger
    logger = set_logger(Path(__file__).name.split('.')[0])
    # check plot status
    plot = args.plot
    if plot and plt is None:
        logger.warning('No plots due to missing package `matplotlib`!')
        plot = False
    # set output path
    p_out = Path(args.output_path)
    # Create the output directory if it didn't already exist.
    p_out.mkdir(parents=True, exist_ok=True)
    # parse L0B
    raw = Raw(hdf5file=args.l0b_file)
    # run noise estimator
    noise_prods = est_noise_power_from_raw(
        raw,
        num_rng_block=args.num_rng_block,
        algorithm=args.algorithm,
        cpi=args.cpi,
        diff=not args.no_diff,
        diff_method=args.diff_method,
        median_ev=not args.no_median_ev,
        dif_quad=args.diff_quad,
        perc_invalid_rngblk=args.pct_invalid_rngblk,
        exclude_first_last=args.exclude_first_last,
        logger=logger
    )
    # report stats of noise power
    for ns_prod in noise_prods:
        logger.info(f'Noise stats for w/ TxRx Pol={ns_prod.txrx_pol} '
                    f'and Band={ns_prod.freq_band} ...')
        _log_noise_stats_db(ns_prod.power_linear, logger)
    # dump noise products to a JSON file
    f_json = Path(args.json_file)
    logger.info(f'Dumping Noise products to -> "{f_json}"')
    with open(f_json, 'w') as fidw:
        json.dump([vars(ns_prod) for ns_prod in noise_prods],
                  fidw, indent=4, cls=JsonNumpyEncoder)
    # plotting
    if plot:
        for nn, ns_prod in enumerate(noise_prods, start=1):
            plot_name = (f'Plot_Noise_Power_{ns_prod.method}_'
                         f'Freq{ns_prod.freq_band}_'
                         f'Pol{ns_prod.txrx_pol}.png')
            plt.figure()
            plt.plot(ns_prod.slant_range * 1e-3,
                     10 * np.log10(ns_prod.power_linear), 'r*--')
            plt.xlabel('Slant Range (km)')
            plt.ylabel(r'Noise Power (${dB_{DN^2}}$)')
            plt.grid(True)
            plt.title(f'{ns_prod.method} Noise Power '
                      fr'within $ENBW={ns_prod.enbw * 1e-6}$(MHz)'
                      f'\nFor TxRx-Pol={ns_prod.txrx_pol} & '
                      f'Freq-Band={ns_prod.freq_band} ')
            plt.savefig(p_out.joinpath(plot_name))

    logger.info(f'Elapsed time (sec) -> {time.time() - tic:.1f}')


def _log_noise_stats_db(ns_pow_lin: np.ndarray, logger: 'logging.Logger'):
    """Helper function to log noise power stats in dB"""
    p_db = 10 * np.log10(ns_pow_lin)
    logger.info('Noise power (min, max) in (dB) -> '
                f'({np.nanmin(p_db):.3f}, {np.nanmax(p_db):.3f})')
    logger.info('Noise Power (mean, std) (dB) -> '
                f'({np.nanmean(p_db):.3f}, {np.nanstd(p_db):.3f})')


if __name__ == '__main__':
    run_noise_estimator(cmd_line_parse())
