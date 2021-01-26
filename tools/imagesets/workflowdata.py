# yuck!
workflowdata = {
        # remote subdir, also name in local data dir
        "L0B_RRSD_REE1":
        # files to grab
        [
            "REE_L0B_out17.h5",
            "README.txt",
        ],
    
        # and so forth
        "L0B_RRSD_REE2":
        [
            "REE_L0B_NISAR_array144sq_01.h5",
            "REE_ANTPAT_DATA.h5",
            "README.txt",
        ],
    
        "L0B_RRSD_DIST1":
        [
            "REE_L0B_NISAR_DATA_PASS1.h5",
            "REE_ANTPAT_DATA.h5",
            "README.txt",
        ],
    
        "L0B_RRSD_DIST2":
        [
            "REE_L0B_NISAR_DATA_PASS2.h5",
            "REE_ANTPAT_DATA.h5",
            "README.txt",
        ],
    
        "L0B_RRSD_ALPSRP037370690":
        [
            "ALPSRP037370690.L0B.h5",  
            "README.txt",
        ],
    
        "L0B_RRSD_ALPSRP271200680":
        [
            "ALPSRP271200680.L0B.h5",  
            "README.txt",
        ],
    
        "L0B_RRSD_REE_CALTOOL":
        [
            "L0B_RRSD_REE_CALTOOL.h5", 
            "README.txt",
        ],
    
        "L1_RSLC_S1B_IW_SLC__1SDV_20180504T104507_20180504T104535_010770_013AEE_919F":
        [
            "S1B_IW_SLC__1SDV_20180504T104507_20180504T104535_010770_013AEE_919F.h5",
            "nisar-dem/EPSG32718/EPSG32718.vrt",
            "nisar-dem/EPSG32718/N9000E0000.tif",
            "nisar-dem/EPSG32718/N9000E0200.tif",
            "nisar-dem/EPSG32718/N9000E0400.tif",
            "nisar-dem/EPSG32718/N9200E0000.tif",
            "nisar-dem/EPSG32718/N9200E0200.tif",
            "nisar-dem/EPSG32718/N9200E0400.tif",
            "nisar-dem/EPSG32718/N9400E0000.tif",
            "nisar-dem/EPSG32718/N9400E0200.tif",
            "nisar-dem/EPSG32718/N9400E0400.tif",    
            "README.txt",
        ],
    
        "L1_RSLC_UAVSAR_SanAnd_05024_18038_006_180730_L090_CX_129_05":
        [
            "SanAnd_05024_18038_006_180730_L090_CX_129_05.h5",
            "nisar-dem/EPSG32610/EPSG32610.vrt",
            "nisar-dem/EPSG32610/N4000E0400.tif",
            "nisar-dem/EPSG32610/N4000E0600.tif",
            "README.txt",
        ],
    
        "L1_RSLC_UAVSAR_SanAnd_05518_12018_000_120419_L090_CX_143_03":
        [
            "SanAnd_05518_12018_000_120419_L090_CX_143_03.h5",
            "dem.tif",
            "README.txt",
        ],
 
        "L1_RSLC_UAVSAR_SanAnd_05518_12128_008_121105_L090_CX_143_02":
        [
            "SanAnd_05518_12128_008_121105_L090_CX_143_02.h5",
            "dem.tif",
            "README.txt",
        ],
 
        "L1_RSLC_UAVSAR_NISARP_32039_19049_005_190717_L090_CX_129_03":
        [
            "NISARP_32039_19049_005_190717_L090_CX_129_03.h5",
            "nisar-dem/EPSG32617/EPSG32617.vrt",
            "nisar-dem/EPSG32617/N3800E0200.tif",
            "README.txt",
        ],
    
        "L1_RSLC_UAVSAR_NISARP_32039_19052_004_190726_L090_CX_129_02":
        [
            "NISARP_32039_19052_004_190726_L090_CX_129_02.h5",
            "nisar-dem/EPSG32617/EPSG32617.vrt",
            "nisar-dem/EPSG32617/N3800E0200.tif",
            "README.txt",
        ],
    }

# dictionaries definining mappig of workflow tests to data
# each key is the test name, value is the corresponding data
workflowtests = {
    'rslc': {"rslc_" + name: "L0B_RRSD_" + name for name in [
        "REE1",
        "REE2",
        "DIST1",
        "DIST2",
        "ALPSRP037370690",
        "ALPSRP271200680",
    ]},
        
    'gslc': {"gslc_" + name: "L1_RSLC_" + name for name in [
        "UAVSAR_SanAnd_05024_18038_006_180730_L090_CX_129_05",
        "UAVSAR_NISARP_32039_19049_005_190717_L090_CX_129_03",
        "UAVSAR_NISARP_32039_19052_004_190726_L090_CX_129_02",
        "UAVSAR_SanAnd_05518_12018_000_120419_L090_CX_143_03",
        "UAVSAR_SanAnd_05518_12128_008_121105_L090_CX_143_02",
    ]},
    
    'gcov': {"gcov_" + name: "L1_RSLC_" + name for name in [
        "UAVSAR_SanAnd_05024_18038_006_180730_L090_CX_129_05",
        "UAVSAR_NISARP_32039_19049_005_190717_L090_CX_129_03",
        "UAVSAR_NISARP_32039_19049_005_190717_L090_CX_129_03",
        "S1B_IW_SLC__1SDV_20180504T104507_20180504T104535_010770_013AEE_919F",
        "UAVSAR_SanAnd_05518_12018_000_120419_L090_CX_143_03",
        "UAVSAR_SanAnd_05518_12128_008_121105_L090_CX_143_02",
    ]},
    
    'insar': {
        "insar_UAVSAR_NISARP_32039_19049-005_19052-004_129":
            [
                "L1_RSLC_UAVSAR_NISARP_32039_19049_005_190717_L090_CX_129_03",
                "L1_RSLC_UAVSAR_NISARP_32039_19052_004_190726_L090_CX_129_02",
            ],
        "insar_UAVSAR_SanAnd_05518_12018-000_12128-008_143": 
            [
                "L1_RSLC_UAVSAR_SanAnd_05518_12018_000_120419_L090_CX_143_03",
                "L1_RSLC_UAVSAR_SanAnd_05518_12128_008_121105_L090_CX_143_02",
            ],
    },
    
    'noisest': {"noisest_" + name: "L0B_RRSD_" + name for name in [
        "REE_CALTOOL",
    ]},
}    
