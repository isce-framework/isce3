set (PXDS DEMInterpolator.pxd
          Geo2rdr.pxd
          Geocode.pxd
          geometry.pxd
          pyGeo2rdr.pxd
          pyGeocode.pxd
          pyTopo.pxd
          SerializeGeometry.pxd
          Topo.pxd)

add_isce_cythondir(geometry "${PXDS}")

