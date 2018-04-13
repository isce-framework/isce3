
# Copy files used for various tests
file(COPY envisat.slc DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY envisat.slc.vrt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY warped_envisat.slc DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY warped_envisat.slc.vrt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY offsets/range.off DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)
file(COPY offsets/range.off.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)
file(COPY offsets/azimuth.off DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)
file(COPY offsets/azimuth.off.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/offsets)

# end of file
