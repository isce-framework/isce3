import time

import journal

from pybind_nisar.workflows import h5_prep, rdr2geo, geo2rdr, resample_slc
from pybind_nisar.workflows.yaml_argparse import YamlArgparse
from pybind_nisar.workflows.insar_runconfig import InsarRunConfig


def run(cfg):
    '''
    Run INSAR workflow with parameters in cfg dictionary
    '''
    info_channel = journal.info("insar.run")
    info_channel.log("starting INSAR")

    t_all = time.time()

    rdr2geo.run(cfg)
    geo2rdr.run(cfg)
    resample_slc.run(cfg)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran partial INSAR in {t_all_elapsed:.3f} seconds")


if __name__=="__main__":
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()
    insar_runcfg = InsarRunConfig(args)
    h5_prep.run(insar_runcfg.cfg, 'GUNW')
    run(insar_runcfg.cfg)
