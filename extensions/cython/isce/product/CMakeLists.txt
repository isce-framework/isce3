set (PXDS Metadata.pxd
          ProcessingInformation.pxd
          Product.pxd
          pyMetadata.pxd
          pyProcessingInformation.pxd
          pyProduct.pxd
          pyRadarGridParameters.pxd
          pySwath.pxd
          RadarGridParameters.pxd
          Swath.pxd)

add_isce_cythondir(product "${PXDS}")

