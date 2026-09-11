# Configuration Guide

This is a guide for all the options for configuration of the cmd line and script version of running the pipeline

## Options for the commandline tool

```bash
usage: run_stack.py --slc-folder SLC_FOLDER --out-folder OUT_FOLDER
                    --dem-file DEM_FILE [--pair-level PAIR_LEVEL]
                    [--freq FREQ] [--pol POL] [--overwrite] [--debug]
                    [--steps {coarse,dense,invert,rubbersheet,resample}
                             [{coarse,dense,invert,rubbersheet,resample} ...]]

Run NISAR StackProcessor pipeline.

options:

  --slc-folder SLC_FOLDER
                        Folder containing NISAR RSLC HDF5 files.

  --out-folder OUT_FOLDER
                        Output stack folder.

  --dem-file DEM_FILE   DEM file used by the stack processing workflow.

  --pair-level PAIR_LEVEL
                        Dense-offset pair level. Default: 2.

  --freq FREQ           Frequency to process. Default: A.

  --pol POL             Polarization to process. Default: HH.

  --overwrite           Overwrite existing outputs.

  --steps {coarse,dense,invert,rubbersheet,resample} ...
                        Pipeline stages to run. If omitted, all stages run in
                        this order: coarse dense invert rubbersheet resample.
```



## Options for the script pipeline

The optional inputs above can also be supplied to the script version of the pipeline:

```python
processor = self = StackProcessor(
    slc_folder='<folder of L1 RSLC H5 files to be aligned>',
    out_folder='<folder for stack output>',
    dem_file='<path to DEM file *.dem.wgs84>',

    # Optional arguments
    pair_level=2,         # dense-offset pairing level
                          # 1 = adjacent acquisitions only
                          # 2 = first- and second-neighbor pairs
    freq='A',             # NISAR frequency
    pol='HH',             # polarization
    overwrite=False,      # overwrite existing outputs
)
```

the ```rdr2geo``` ,```dense_offsets``` and```rubbersheet``` steps of the workflow can be configured with isce3 format. The default is as defined in nisar's example [insar.yaml](https://github.com/isce-framework/isce3/blob/develop/share/nisar/defaults/insar.yaml).

example

```python
from stack_utils import load_config

runcfg = load_config('<path_to_yaml>')
cfg = runcfg.cfg["runconfig"]["groups"]
dense_cfg = cfg["processing"]["dense_offsets"]
rdr2geo_cfg = cfg['processing']['rdr2geo']
rubbersheet_cfg = cfg['processing']['rubbersheet']

processor.coarse_register_scans(rdr2geo_cfg=rdr2geo_cfg)
processor.dense_offset_pairs(dense_cfg=dense_cfg)
processor.rubbersheet(rubbersheet_cfg=rubbersheet_cfg)
```
