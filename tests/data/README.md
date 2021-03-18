# Test Data

This README gives brief descriptions on the generation of various test data
found in this directory.

## Winnipeg

- The original product from which the test data are generated: winnip_31604_12061_004_120717_L090_CX_129_05.h5
  This is a UAVSAR NISAR simulated product. Check science/LSAR/identification/productVersion to get the product version
  As for NISAR data/ Simulated NISAR data have to be intended as zero-doppler

- All winnipeg-related "reference" products in this folder (e.g. warped_winnipeg.slc) have been generated using ISCE2 v2.3.
  The ISCE2 processing uses "winnip_31604_12061_004_120717_L090_CX_129_05.h5" as reference and secondary SLC to create the golden/reference dataset.

## Attitude

Attitude sample data in NISAR format (see JPL D-102264) was provided by the
G&C team (Dave Bates, et al) during NASA Internal Thread Test 04 (NITT-04)
activities in late 2020/early 2021. The hash of the original file delivered
by email is

```
$ md5sum NISAR_ANC_L_PR_FRP_20210114T023357_20200107T200000_20200109T040000.xml
2101bddf088d3b4e8e0e3051931f8284  NISAR_ANC_L_PR_FRP_20210114T023357_20200107T200000_20200109T040000.xml
```

To reduce the size of the test data, this file was trimmed to the first ten
records to produce `attitude.xml`.

## Orbit

Orbit sample data in NISAR format (see JPL D-102253) provided by Paul Ries
via email on 2020-07-06. It was generated from Jason2 data. The hash of the
original file is

```
$ md5sum -b smoothFinal.xml.gz
f415fc38e1feff0bb1453782be3d2b5f *smoothFinal.xml.gz
```

This file was uncompressed and trimmed to the first ten state vectors in
order to reduce the size of the `orbit.xml` file stored here.
