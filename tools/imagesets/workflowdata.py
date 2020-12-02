# yuck!
workflowdata = [
    (
        # local subdir (i.e. the one used for local testing)
        "test_rslc",
        # remote subdir (will be stripped in destination)
        "RSLC_test_REE1",
        # files to grab
        [
            "run_config_rslc.yaml",
            "input/REE_L0B_out17.h5",
        ],
    ),
    # and so forth
    (
        "test_gslc",
        "GSLC_GCOV_test_SanAnd",
        [
            "run_config_gslc_v2.yaml",
            "input/SanAnd_05024_18038_006_180730_L090_CX_129_05.h5",
            "input/nisar-dem/EPSG32610/EPSG32610.vrt",
            "input/nisar-dem/EPSG32610/N4000E0400.tif",
            "input/nisar-dem/EPSG32610/N4000E0600.tif",
        ],
    ),
    (
        "test_gcov",
        "GSLC_GCOV_test_SanAnd",
        [
            "run_config_gcov_v3.yaml",
            "input/SanAnd_05024_18038_006_180730_L090_CX_129_05.h5",
            "input/nisar-dem/EPSG32610/EPSG32610.vrt",
            "input/nisar-dem/EPSG32610/N4000E0400.tif",
            "input/nisar-dem/EPSG32610/N4000E0600.tif",
        ],
    ),
]
