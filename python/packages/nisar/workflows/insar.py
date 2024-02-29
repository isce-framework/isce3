#!/usr/bin/env python3
import time

import journal
from nisar.workflows import (bandpass_insar, crossmul,
                             dense_offsets, geo2rdr,geocode_insar,
                             h5_prep, filter_interferogram,
                             offsets_product, prepare_insar_hdf5, rdr2geo,
                             resample_slc, rubbersheet,
                             split_spectrum, unwrap, ionosphere, baseline,
                             troposphere, solid_earth_tides)

from nisar.workflows.geocode_insar import InputProduct
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.persistence import Persistence
from nisar.workflows.yaml_argparse import YamlArgparse


def run(cfg: dict, out_paths: dict, run_steps: dict):
    '''
    Run INSAR workflow with parameters in cfg dictionary
    '''
    info_channel = journal.info("insar.run")
    info_channel.log("starting INSAR")

    t_all = time.time()

    if run_steps['bandpass_insar']:
        bandpass_insar.run(cfg)

    if run_steps['prepare_insar_hdf5']:
        prepare_insar_hdf5.run(cfg)

    if run_steps['rdr2geo']:
        rdr2geo.run(cfg)

    if run_steps['geo2rdr']:
        geo2rdr.run(cfg)

    if run_steps['coarse_resample']:
        resample_slc.run(cfg, 'coarse')

    if (run_steps['dense_offsets']) and \
            (cfg['processing']['dense_offsets']['enabled']):
        dense_offsets.run(cfg)

    if (run_steps['offsets_product']) and \
            (cfg['processing']['offsets_product']['enabled']):
        offsets_product.run(cfg, out_paths['ROFF'])

    if run_steps['rubbersheet'] and \
            cfg['processing']['rubbersheet']['enabled'] and \
            'RIFG' in out_paths:
        rubbersheet.run(cfg, out_paths['RIFG'])

    # If enabled, run fine_resampling
    if run_steps['fine_resample'] and \
            cfg['processing']['fine_resample']['enabled'] and \
            'RIFG' in out_paths:
        resample_slc.run(cfg, 'fine')


    # If fine_resampling is enabled, use fine-coregistered SLC
    # to run crossmul
    if run_steps['crossmul'] and 'RIFG' in out_paths:
        if cfg['processing']['fine_resample']['enabled']:
            crossmul.run(cfg, out_paths['RIFG'], 'fine')
        else:
            crossmul.run(cfg, out_paths['RIFG'], 'coarse')

    # Run insar_filter only
    if run_steps['filter_interferogram'] and \
        cfg['processing']['filter_interferogram']['filter_type'] != 'no_filter' and \
            'RIFG' in out_paths:
        filter_interferogram.run(cfg, out_paths['RIFG'])

    if run_steps['unwrap'] and 'RUNW' in out_paths:
        unwrap.run(cfg, out_paths['RIFG'], out_paths['RUNW'])

    if run_steps['ionosphere'] and \
            cfg['processing']['ionosphere_phase_correction']['enabled'] and \
            'RUNW' in out_paths:
        split_spectrum.run(cfg)
        ionosphere.run(cfg, out_paths['RUNW'])

    if run_steps['geocode'] and 'GUNW' in out_paths:
        # Geocode RIFG
        geocode_insar.run(cfg, out_paths['RIFG'], out_paths['GUNW'], InputProduct.RIFG)
        # Geocode RUNW
        geocode_insar.run(cfg, out_paths['RUNW'], out_paths['GUNW'], InputProduct.RUNW)

    if run_steps['geocode'] and 'GOFF' in out_paths:
        # Geocode ROFF
        geocode_insar.run(cfg, out_paths['ROFF'], out_paths['GOFF'], InputProduct.ROFF)

    if 'GUNW' in out_paths and run_steps['troposphere'] and \
            cfg['processing']['troposphere_delay']['enabled']:
        troposphere.run(cfg, out_paths['GUNW'])

    if 'GUNW' in out_paths and run_steps['solid_earth_tides']:
        solid_earth_tides.run(cfg, out_paths['GUNW'])

    if run_steps['baseline']:
        baseline.run(cfg, out_paths)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran INSAR in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    insar_runcfg = InsarRunConfig(args)

    # To allow persistence, a logfile is required. Raise exception
    # if logfile is None and persistence is requested
    logfile_path = insar_runcfg.cfg['logging']['path']
    if (logfile_path is None) and insar_runcfg.args.restart:
        raise ValueError('InSAR workflow persistence requires to specify a logfile')
    persist = Persistence(logfile_path, insar_runcfg.args.restart)

    # run InSAR workflow
    if persist.run:
        _, out_paths = h5_prep.get_products_and_paths(insar_runcfg.cfg)

        run(insar_runcfg.cfg, out_paths, persist.run_steps)
