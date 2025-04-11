"""
Prepare InSAR HDF5 for
GUNW, GOFF, RIFG, ROFF, and RUNW
"""
import journal
from nisar.products.insar import (GOFFWriter, GUNWWriter, RIFGWriter,
                                  ROFFWriter, RUNWWriter)
from nisar.workflows.h5_prep import get_products_and_paths

def prepare_insar_hdf5(cfg, output_hdf5, dst):
    """
    Prepare InSAR (GOFF, GUNW, RIFG, ROFF,  RUNW) HDF5 products.

    Parameters
    ----------
    cfg: dict
        runconfig dictionary
    output_hdf5: str
        the output path of the InSAR product
    dst : str
        the name of the InSAR product
    """

    if "RUNW" in dst:
        with RUNWWriter(name = output_hdf5, mode = 'w',
                        runconfig_dict = cfg,
                        runconfig_path="None") as runw:
            runw.save_to_hdf5()
    if "ROFF" in dst:
        with ROFFWriter(name = output_hdf5, mode = 'w',
                        runconfig_dict = cfg,
                        runconfig_path="None") as roff:
            roff.save_to_hdf5()
    if "RIFG" in dst:
        with RIFGWriter(name = output_hdf5, mode = 'w',
                        runconfig_dict = cfg,
                        runconfig_path="None") as rifg:
            rifg.save_to_hdf5()
    if "GUNW" in dst:
        with GUNWWriter(name = output_hdf5, mode = 'w',
                        runconfig_dict = cfg,
                        runconfig_path="None") as gunw:
            gunw.save_to_hdf5()
    if "GOFF" in dst:
        with GOFFWriter(name = output_hdf5, mode = 'w',
                        runconfig_dict = cfg,
                        runconfig_path="None") as goff:
            goff.save_to_hdf5()


def run(cfg: dict) -> dict:
    """
    prepare datasets
    Returns dict of output path(s); used for InSAR workflow
    """
    info_channel = journal.info("prepare_insar_hdf5.run")
    info_channel.log("preparing InSAR HDF5 products")

    product_dict, h5_paths = get_products_and_paths(cfg)
    for sub_prod_type in product_dict:
        out_path = h5_paths[sub_prod_type]
        prepare_insar_hdf5(cfg, out_path, sub_prod_type)

    info_channel.log("successfully ran prepare_insar_hdf5")

    return h5_paths
