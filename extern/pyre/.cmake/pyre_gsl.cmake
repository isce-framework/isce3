# -*- cmake -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# build the gsl package
function(pyre_gslPackage)
  # if we have gsl
  if(${GSL_FOUND})
    # install the sources straight from the source directory
    install(
      DIRECTORY gsl
      DESTINATION ${PYRE_DEST_PACKAGES}
      FILES_MATCHING PATTERN *.py
      )
  endif()
  # all done
endfunction(pyre_gslPackage)


# build the gsl module
function(pyre_gslModule)
  # if we have gsl
  if (${GSL_FOUND})
    Python3_add_library(gslmodule MODULE)
    # adjust the name to match what python expects
    set_target_properties(gslmodule PROPERTIES LIBRARY_OUTPUT_NAME gsl)
    set_target_properties(gslmodule PROPERTIES SUFFIX ${PYTHON3_SUFFIX})
    # set the include directories
    target_include_directories(gslmodule PRIVATE ${GSL_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
    # set the libraries to link against
    target_link_libraries(
      gslmodule PRIVATE
      ${GSL_LIBRARIES} pyre journal
      )
    # add the sources
    target_sources(gslmodule PRIVATE
      gsl/gsl.cc
      gsl/blas.cc
      gsl/exceptions.cc
      gsl/histogram.cc
      gsl/linalg.cc
      gsl/matrix.cc
      gsl/metadata.cc
      gsl/numpy.cc
      gsl/pdf.cc
      gsl/permutation.cc
      gsl/rng.cc
      gsl/stats.cc
      gsl/vector.cc
      )
    # copy the capsule definitions to the staging area
    file(
      COPY gsl/capsules.h
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../lib/pyre/gsl
      )
    if (${MPI_FOUND})
      # add the MPI aware sources to the pile
      target_sources(gslmodule PRIVATE gsl/partition.cc)
      # the mpi include directories
      target_include_directories(gslmodule PRIVATE ${MPI_CXX_INCLUDE_PATH})
      # and the mpi libraries
      target_link_libraries(gslmodule PRIVATE ${MPI_CXX_LIBRARIES})
    endif()

    # install the extension
    install(
      TARGETS gslmodule
      LIBRARY
      DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/gsl
      )
    # and publish the capsules
    install(
      FILES ${CMAKE_CURRENT_SOURCE_DIR}/gsl/capsules.h
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/pyre/gsl
      )
  endif()
  # all done
endfunction(pyre_gslModule)


# end of file
