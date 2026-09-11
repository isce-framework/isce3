# ISCE3 Stack Processing

A working first draft to align a stack of **NISAR scans**  to a reference.  Currently works for single GPU setups.

## Preperation

1. Regular ISCE DEM files are needed with the following postfix: 

```
<dem_file_name>.dem.wgs84

<dem_file_name>.dem.wgs84.vrt

<dem_file_name>.dem.wgs84.xml
```

   A convenient way to produce them is the  [dem-stitcher tool for isce workflow](https://github.com/ACCESS-Cloud-Based-InSAR/dem-stitcher/blob/dev/notebooks/Staging_a_DEM_for_ISCE2.ipynb).

2. Download the L1 RSLC h5 product for the scans you wish to process and put them in the same folder.  The h5 files should be in the original naming convention:####    

**Example:**

```bash
NISAR_L1_PR_RSLC_004_122_D_067_4005_DHDH_A_20251106T160541_20251106T160622_X05009_N_F_J_001.h5
```

**Example aligned to format:**

```bash
NISAR_L1_PR_RSLC_<Cycle>_<Track>_<OrbitDirection>_<Frame>_<BandwidthMode>_<Polarization>_<Source>_<StartDateTime>_ <EndDateTime> _<CRID>_<Accuracy>_<Coverage>_<ProcessingLocation>_<Counter>.h5
NISAR_L1_PR_RSLC_  004  _  122  _       D        _  067  _      4005     _     DHDH     _   A    _20251106T160541_20251106T160622_X05009_    N     _    F     _         J          _   001   .h5
```



Stack alignment will be performed on all scans for a specific **Track,Frame and Direction*** combination.

If multiple combinations are present, they will be processed as separate stacks.

## Aligning the stacks

Run the following command to start the alignment workflow:

```python
python run_stack.py \
    --slc-folder <folder of l1 RSLC h5s to be aligned> \
    --out-folder <folder of stack output> \
    --dem-file <path to the dem file *.dem.wgs84> 
```

You can also run the workflow from a custom python script as shown below:

```python
from StackProcessor import StackProcessor
processor = self = StackProcessor(
    slc_folder='<folder of l1 RSLC h5s to be aligned>',
    out_folder='<folder of stack output>',
    dem_file='<path to the dem file *.dem.wgs84>',
    )
processor.coarse_register_scans()
processor.dense_offset_pairs()
processor.invert_dense_offset_pairs()
processor.rubbersheet()
processor.resample_slcs()
```

To do this you would need to add the stack folder to python path by doing:

```bash
export ISCE_STACK={full_path_to_your_share/stack}
export PYTHONPATH=${PYTHONPATH}:${ISCE_STACK}
```

For detailed config of the cmd and script way of runing the pipleine, see [configuration guide](configuration-guide.md)

## Alignment Steps

the scans in the stack is aligned using the following steps, again a stack is all the scans found in the input folder with the same track, frame and direction.  By default the ** **earliest scan**  is used as the reference

- coarse_register_scans
  
  - geometry of the reference is extracted using ```isce3.cuda.geometry.Rdr2Geo```
  
  - coarse offsets for each secondary scans are calculated with ```isce3.cuda.geometry.Geo2Rdr```
  
  - secondary h5s are than resampled to coarse aligned slcs with ```nisar.workflows.resample_slc_v2.resample_secondary_rslc_onto_reference```

- dense_offset_pairs
  
  - dense offsets are calculated between date pairs using ```isce3.cuda.matchtemplate.PyCuAmpcor``` with configs provided to ```StackProcessor.dense_offset_pairs```.  See more details about which pairs are choosen [here](./dense-offset-and-inversion.md#about-dense-offset-pairs)

- invert_dense_offset_pairs
  
  - Offsets are then inverted to obtain offset to the reference scan.
    This process prevents files with large offsets from skewing offsets fir the rest of the stack.  You can find more detail [here](./dense-offset-and-inversion.md#dense-offset-inversion)

- rubbersheet
  
  - The inverted offsets are then upscaled to full resoltuion through interpolation

- resample_slcs
  
  - the secondary h5s are then sampled to aligned slcs that can be used for downstream processing with Mintpy or dolphin

## Output Folders

This sections descrbes all the output folders produced by the pipeline.

Each stack is stored in a directory identified by its track, frame, and orbit direction, for example:

```textile
output_folder/
├── processor.pkl # a pickle of the processor object
├── t<TrackNo>_f<FrameNo>_<A/D scan direction>/ 
├── tXX_fXX_A/
├── tXX_fXX_D/
```

A processed stack contains the following directories:

```textile
tXX_fXX_A/
├── ref_geom/          [coarse]      rdr2geo reference radar geometry
├── coarse_offsets/    [coarse]      geo2rdr range/azimuth offsets
├── stack/             [coarse]      coarsely registered SLC stack
├── dense_pairwise/    [dense]       pairwise Ampcor dense offsets
├── inverted_offsets/  [invert]      inverted offsets to reference SLC
├── h5_stack/          [rubbersheet] by product of rubbersheet
├── rubber/            [rubbersheet] final rubbersheeted offsets
├── merged/            [resample]    FINAL aligned/resampled SLC stack
```


