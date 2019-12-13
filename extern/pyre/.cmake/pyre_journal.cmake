# -*- cmake -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# buld the journal pcckage
function(pyre_journalPackage)
  # install the sources straight from the source directory
  install(
    DIRECTORY journal
    DESTINATION ${PYRE_DEST_PACKAGES}
    FILES_MATCHING PATTERN *.py
    )
  # build the package meta-data
  configure_file(
    journal/meta.py.in journal/meta.py
    @ONLY
    )
  # install the generated package meta-data file
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/journal
    DESTINATION ${PYRE_DEST_PACKAGES}
    FILES_MATCHING PATTERN *.py
    )
  # all done
endfunction(pyre_journalPackage)


# build libjournal
function(pyre_journalLib)
  # copy the journal headers over to the staging area
  file(
    COPY journal
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/pyre
    FILES_MATCHING
    PATTERN *.h PATTERN *.icc
    PATTERN journal/journal.h EXCLUDE
    )
  # and the journal master header within the pyre directory
  file(
    COPY journal/journal.h
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/pyre
    )

  # the libjournal target
  add_library(journal SHARED)
  # define the core macro
  set_target_properties(journal PROPERTIES COMPILE_DEFINITIONS PYRE_CORE)
  # set the include directories
  target_include_directories(
    journal PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    )
  # add the sources
  target_sources(journal
    PRIVATE
    journal/Chronicler.cc
    journal/Console.cc
    journal/Device.cc
    journal/Renderer.cc
    journal/Streaming.cc
    journal/debuginfo.cc
    journal/firewalls.cc
    journal/journal.cc
    )

  # libpyre and libjournal
  install(
    TARGETS journal
    LIBRARY DESTINATION lib
    )

  # all done
endfunction(pyre_journalLib)


# build the journal python extension
function(pyre_journalModule)
  # journal
  Python3_add_library(journalmodule MODULE)
  # turn on the core macro
  set_target_properties(journalmodule PROPERTIES COMPILE_DEFINITIONS PYRE_CORE)
  # adjust the name to match what python expects
  set_target_properties(journalmodule PROPERTIES LIBRARY_OUTPUT_NAME journal)
  set_target_properties(journalmodule PROPERTIES SUFFIX ${PYTHON3_SUFFIX})
  # set the libraries to link against
  target_link_libraries(journalmodule PRIVATE journal)
  # add the sources
  target_sources(journalmodule PRIVATE
    journal/journal.cc
    journal/DeviceProxy.cc
    journal/channels.cc
    journal/exceptions.cc
    journal/init.cc
    journal/metadata.cc
    journal/tests.cc
    )
  # install
  install(
    TARGETS journalmodule
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/journal
    )
endfunction(pyre_journalModule)


# end of file
