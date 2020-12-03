# yuck!
workflowdata = [
    (
        # remote subdir, also name in local data dir
        "L0B_RRSD_REE1",
        # files to grab
        [
            "REE_L0B_out17.h5",
            "README.txt",
        ],
    ),
    # and so forth
    (
        "L1_RSLC_UAVSAR_SanAnd1",
        [
            "SanAnd_05024_18038_006_180730_L090_CX_129_05.h5",
            "nisar-dem/EPSG32610/EPSG32610.vrt",
            "nisar-dem/EPSG32610/N4000E0400.tif",
            "nisar-dem/EPSG32610/N4000E0600.tif",
            "README.txt",
        ],
    ),
]

# dictionaries definining mappig of workflow tests to data
# each key is the test name, value is the corresponding data
rslctestdict = {
    "RSLC_REE1": "L0B_RRSD_REE1",
    }
    
gslctestdict = {
    "GSLC_UAVSAR_SanAnd1": "L1_RSLC_UAVSAR_SanAnd1",
    }

gcovtestdict = {
    "GCOV_UAVSAR_SanAnd1": "L1_RSLC_UAVSAR_SanAnd1",
    }
