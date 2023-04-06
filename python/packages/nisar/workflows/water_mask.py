import h5py
import journal
from osgeo import gdal
import time
import numpy as np 

from nisar.workflows import h5_prep
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse


def project_water_to_geogrid(input_water_path, geogrid):

    inputraster = gdal.Open(input_water_path)
    output_extent = (geogrid.start_x,
                     geogrid.start_y + geogrid.length * geogrid.spacing_y,
                     geogrid.start_x + geogrid.width * geogrid.spacing_x,
                     geogrid.start_y)

    gdalwarp_options = gdal.WarpOptions(format="MEM",
                                        dstSRS=f"EPSG:{geogrid.epsg}",
                                        xRes=geogrid.spacing_x,
                                        yRes=np.abs(geogrid.spacing_y),
                                        resampleAlg='near',
                                        outputBounds=output_extent)
    dst_ds = gdal.Warp("", inputraster, options=gdalwarp_options)

    projected_data = dst_ds.ReadAsArray()

    return projected_data

def run(cfg, gunw_hdf5: str):
    info_channel = journal.info("water_mask.run")
    info_channel.log("starting water mask for GUNW")
    t_all = time.time()

    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    water_mask_path = cfg['dynamic_ancillary_file_group']['water_mask_file']
    geogrids = cfg['processing']['geocode']['geogrids']

    if water_mask_path is not None:
        with h5py.File(gunw_hdf5, 'a', libver='latest', swmr=True) as src_gunw:
            for freq in freq_pols.keys():
                freq_path = f'/science/LSAR/GUNW/grids/frequency{freq}'
                water_mask_h5_path = f'{freq_path}/interferogram/waterMask'
                geogrid = geogrids[freq]
                water_mask = project_water_to_geogrid(water_mask_path, geogrid)
                water_mask_interpret = np.array(water_mask != 0, dtype='uint8')
                src_gunw[water_mask_h5_path].write_direct(water_mask_interpret)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran water mask in {t_all_elapsed:.3f} seconds")

if __name__ == "__main__":

    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    insar_runcfg = InsarRunConfig(args)
    _, out_paths = h5_prep.get_products_and_paths(insar_runcfg.cfg)
    run(insar_runcfg.cfg, gunw_hdf5 = out_paths['GUNW'])

