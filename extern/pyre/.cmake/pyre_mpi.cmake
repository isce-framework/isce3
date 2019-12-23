# -*- cmake -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


function(pyre_mpiPackage)
  # if we have mpi
  if(${MPI_FOUND})
    # install the sources straight from the source directory
    install(
      DIRECTORY mpi
      DESTINATION ${PYRE_DEST_PACKAGES}
      FILES_MATCHING PATTERN *.py
      )
    # build the package meta-data
    configure_file(
      mpi/meta.py.in mpi/meta.py
      @ONLY
      )
    # install the generated package meta-data file
    install(
      DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mpi
      DESTINATION ${PYRE_DEST_PACKAGES}
      FILES_MATCHING PATTERN *.py
      )
  endif()
  # all done
endfunction(pyre_mpiPackage)


# the pyre mpi headers
function(pyre_mpiLib)
  # if we have mpi
  if(MPI_FOUND)
    # copy the mpi headers
    file(
      COPY mpi
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/pyre
      FILES_MATCHING
      PATTERN *.h PATTERN *.icc
      PATTERN mpi/mpi.h EXCLUDE
      )
    # and the mpi master header with the pyre directory
    file(
      COPY mpi/mpi.h
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/pyre
      )
  endif(MPI_FOUND)
  # all done
endfunction(pyre_mpiLib)


# build the mpi module
function(pyre_mpiModule)
  # if we have mpi
  if (${MPI_FOUND})
    Python3_add_library(mpimodule MODULE)
    # adjust the name to match what python expects
    set_target_properties(mpimodule PROPERTIES LIBRARY_OUTPUT_NAME mpi)
    set_target_properties(mpimodule PROPERTIES SUFFIX ${PYTHON3_SUFFIX})
    # set the include directories
    target_include_directories(mpimodule PRIVATE ${MPI_CXX_INCLUDE_PATH})
    # set the libraries to link against
    target_link_libraries(
      mpimodule PRIVATE
      ${MPI_CXX_LIBRARIES} pyre journal
      )
    # add the sources
    target_sources(mpimodule PRIVATE
      mpi/mpi.cc
      mpi/communicators.cc
      mpi/exceptions.cc
      mpi/groups.cc
      mpi/metadata.cc
      mpi/ports.cc
      mpi/startup.cc
      )
    # copy the capsule definitions to the staging area
    file(
      COPY mpi/capsules.h
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../lib/pyre/mpi
      )
    # install the extension
    install(
      TARGETS mpimodule
      LIBRARY
      DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/mpi
      )
    # and publish the capsules
    install(
      FILES ${CMAKE_CURRENT_SOURCE_DIR}/mpi/capsules.h
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/pyre/mpi
      )
  endif()
  # all done
endfunction(pyre_mpiModule)


# end of file
