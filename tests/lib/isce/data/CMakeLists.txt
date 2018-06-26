
# Copy files used for various tests
file(COPY envisat.slc DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY envisat.slc.vrt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY warped_envisat.slc DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY warped_envisat.slc.vrt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY topo.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY srtm_cropped.tif DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY offsets/range.off DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)
file(COPY offsets/range.off.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)
file(COPY offsets/azimuth.off DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)
file(COPY offsets/azimuth.off.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)

# Copy files necessary for topo comparison
file(COPY topo/topo.vrt DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/x.rdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/x.hdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/y.rdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/y.hdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/z.rdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/z.hdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/inc.rdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/inc.hdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/hdg.rdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/hdg.hdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/localInc.rdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/localInc.hdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/localPsi.rdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)
file(COPY topo/localPsi.hdr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/topo)

# end of file
