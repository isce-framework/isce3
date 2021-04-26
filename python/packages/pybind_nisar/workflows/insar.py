#!/usr/bin/env python3
import time

import journal
from pybind_nisar.workflows import (crossmul, dense_offsets, geo2rdr,
                                    geocode_insar, h5_prep, rdr2geo,
                                    resample_slc, unwrap)
from pybind_nisar.workflows.insar_runconfig import InsarRunConfig
from pybind_nisar.workflows.persistence import Persistence
from pybind_nisar.workflows.yaml_argparse import YamlArgparse


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

    if run_steps['resample']:
        resample_slc.run(cfg)

    if (run_steps['dense_offsets']) and \
            (cfg['processing']['dense_offsets']['enabled']):
        dense_offsets.run(cfg)

    if run_steps['crossmul']:
        crossmul.run(cfg, out_paths['RIFG'])

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
    persist = Persistence(args.restart)

    # run InSAR workflow
    if persist.run:
        # prepare HDF5 if needed
        if persist.run_steps['h5_prep']:
            out_paths = h5_prep.run(insar_runcfg.cfg)
        else:
            _, out_paths = h5_prep.get_products_and_paths(insar_runcfg.cfg)

        run(insar_runcfg.cfg, out_paths, persist.run_steps)
