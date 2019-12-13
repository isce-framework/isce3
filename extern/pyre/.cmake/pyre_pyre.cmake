# -*- cmake -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# generate the portinfo file
function(pyre_portinfo)
  # inject the platform information and move it to the stagig area
  configure_file(
    portinfo.in portinfo
    @ONLY
    )
  # install the portinfo file
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/portinfo
    DESTINATION ${PYRE_DEST_INCLUDE}
    )
  # all done
endfunction(pyre_portinfo)


# build the pyre package
function(pyre_pyrePackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY pyre
    DESTINATION ${PYRE_DEST_PACKAGES}
    FILES_MATCHING PATTERN *.py
    )
  # build the package meta-data
  configure_file(
    pyre/meta.py.in pyre/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pyre
    DESTINATION ${PYRE_DEST_PACKAGES}
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(pyre_pyrePackage)


# buld the pyre libraries
function(pyre_pyreLib)
  # buld the libpyre version file
  configure_file(
    pyre/version.cc.in pyre/version.cc
    @ONLY
    )
  # copy the pyre headers over to the staging area
  file(
    COPY pyre
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILES_MATCHING PATTERN *.h PATTERN *.icc
    )

  # the libpyre target
  add_library(pyre SHARED)
  # set the include directories
  target_include_directories(
    pyre PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    )
  # add the sources
  target_sources(pyre
    PRIVATE
    pyre/memory/MemoryMap.cc
    pyre/timers/Display.cc
    pyre/timers/Timer.cc
    ${CMAKE_CURRENT_BINARY_DIR}/pyre/version.cc
    )
  # and the link dependencies
  target_link_libraries(
    pyre
    journal
    )

  # install all the pyre headers
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pyre
    DESTINATION ${PYRE_DEST_INCLUDE}
    FILES_MATCHING PATTERN *.h PATTERN *.icc
    )

  # libpyre and libjournal
  install(
    TARGETS pyre
    LIBRARY DESTINATION lib
    )

  # all done
endfunction(pyre_pyreLib)


# build the pyre extension modules
function(pyre_pyreModule)
  # host
  Python3_add_library(hostmodule MODULE)
  # adjust the name to match what python expects
  set_target_properties(hostmodule PROPERTIES LIBRARY_OUTPUT_NAME host)
  set_target_properties(hostmodule PROPERTIES SUFFIX ${PYTHON3_SUFFIX})
  # set the libraries to link against
  target_link_libraries(hostmodule PRIVATE pyre journal)
  # add the sources
  target_sources(hostmodule PRIVATE
    host/host.cc
    host/cpu.cc
    host/metadata.cc
    )

  # timers
  Python3_add_library(timersmodule MODULE)
  # adjust the name to match what python expects
  set_target_properties(timersmodule PROPERTIES LIBRARY_OUTPUT_NAME timers)
  set_target_properties(timersmodule PROPERTIES SUFFIX ${PYTHON3_SUFFIX})
  # set the libraries to link against
  target_link_libraries(timersmodule PRIVATE pyre journal)
  # add the sources
  target_sources(timersmodule PRIVATE
    timers/timers.cc
    timers/display.cc
    timers/metadata.cc
    )

  # install the pyre extensions
  install(
    TARGETS hostmodule timersmodule
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/pyre/extensions
    )
endfunction(pyre_pyreModule)


# the scripts
function(pyre_pyreBin)
  # the pyre enhanced interpreter
  add_executable(python.pyre)
  # its sources
  target_sources(python.pyre PRIVATE
    python.cc
    )
  # and libraries
  target_link_libraries(python.pyre Python3::Python)

  # install the custom python
  install(
    TARGETS python.pyre
    RUNTIME
    DESTINATION bin
    )
  # install the scripts
  install(
    PROGRAMS pyre pyre-config merlin smith.pyre
    DESTINATION bin
    )
  # all done
endfunction(pyre_pyreBin)


# the configuration files
function(pyre_pyreDefaults)
  # install the configuration files
  install(
    FILES pyre.pfg merlin.pfg
    DESTINATION ${CMAKE_INSTALL_PREFIX}/defaults
  )
  # all done
endfunction(pyre_pyreDefaults)


# end of file
