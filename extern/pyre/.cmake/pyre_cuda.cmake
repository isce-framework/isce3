# -*- cmake -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


function(pyre_cudaPackage)
  # if the user requested CUDA support
  if(WITH_CUDA)
    # install the sources straight from the source directory
    install(
      DIRECTORY cuda
      DESTINATION ${PYRE_DEST_PACKAGES}
      FILES_MATCHING PATTERN *.py
      )
  endif()
  # all done
endfunction(pyre_cudaPackage)


# end of file
