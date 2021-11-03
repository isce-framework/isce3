#!/usr/bin/env python3
import time

import journal
from nisar.workflows import (crossmul, dense_offsets, geo2rdr,
                                    geocode_insar, h5_prep, filter_interferogram,
                                    rdr2geo, resample_slc, rubbersheet, unwrap)
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

    if run_steps['rdr2geo']:
        rdr2geo.run(cfg)

    if run_steps['geo2rdr']:
        geo2rdr.run(cfg)

    if run_steps['coarse_resample']:
        resample_slc.run(cfg, 'coarse')

    if (run_steps['dense_offsets']) and \
            (cfg['processing']['dense_offsets']['enabled']):
        dense_offsets.run(cfg)

    if run_steps['rubbersheet'] and \
            cfg['processing']['rubbersheet']['enabled']:
        rubbersheet.run(cfg, out_paths['RIFG'])

    # If enabled, run fine_resampling
    if run_steps['fine_resample'] and \
            cfg['processing']['fine_resample']['enabled']:
        resample_slc.run(cfg, 'fine')

    # If fine_resampling is enabled, use fine-coregistered SLC
    # to run crossmul
    if run_steps['crossmul']:
        if cfg['processing']['fine_resample']['enabled']:
            crossmul.run(cfg, out_paths['RIFG'], 'fine')
        else:
            crossmul.run(cfg, out_paths['RIFG'], 'coarse')

    # Run insar_filter only
    if run_steps['filter_interferogram'] and \
        cfg['processing']['filter_interferogram']['filter_type'] != 'no_filter':
        filter_interferogram.run(cfg, out_paths['RIFG'])

    if run_steps['unwrap'] and 'RUNW' in out_paths:
        unwrap.run(cfg, out_paths['RIFG'], out_paths['RUNW'])

    if run_steps['geocode'] and 'GUNW' in out_paths:
        geocode_insar.run(cfg, out_paths['RUNW'], out_paths['GUNW'])

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran INSAR in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    insar_runcfg = InsarRunConfig(args)

    # determine what steps if any need to be rerun
    persist = Persistence(insar_runcfg.args.restart)

    # run InSAR workflow
    if persist.run:
        # prepare HDF5 if needed
        if persist.run_steps['h5_prep']:
            out_paths = h5_prep.run(insar_runcfg.cfg)
        else:
            _, out_paths = h5_prep.get_products_and_paths(insar_runcfg.cfg)

        run(insar_runcfg.cfg, out_paths, persist.run_steps)
