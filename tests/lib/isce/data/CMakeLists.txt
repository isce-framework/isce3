#Input files for tests
set (INPUT_SRCS envisat.h5
                warped_envisat.slc
                warped_envisat.slc.vrt
                topo.xml
                srtm_cropped.tif
                offsets/range.off
                offsets/range.off.xml
                offsets/azimuth.off
                offsets/azimuth.off.xml
                rtc/rtc
                rtc/rtc.vrt
                topo/topo.vrt
                topo/x.rdr
                topo/x.hdr
                topo/y.rdr
                topo/y.hdr
                topo/z.rdr
                topo/z.hdr
                topo/inc.rdr
                topo/inc.hdr
                topo/hdg.rdr
                topo/hdg.hdr
                topo/localInc.rdr
                topo/localInc.hdr
                topo/localPsi.rdr
                topo/localPsi.hdr)

add_custom_target(geom_test_data)
foreach (DATASRC ${INPUT_SRCS})
    get_filename_component(RELDIR_PATH ${DATASRC} DIRECTORY)
    add_custom_command(TARGET geom_test_data PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E
                       copy  ${CMAKE_CURRENT_SOURCE_DIR}/${DATASRC} ${CMAKE_CURRENT_BINARY_DIR}/${DATASRC})
endforeach()

# end of file
