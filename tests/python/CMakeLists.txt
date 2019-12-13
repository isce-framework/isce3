# adjust the PYTHONPATH to point to the freshly built packages
set(PYTEST_PYTHONPATH
  ${CMAKE_INSTALL_PREFIX}/packages:${ISCE_BUILDPACKAGESDIR}:$ENV{PYTHONPATH}
  CACHE INTERNAL "python paths for pytest")

set(PYTEST_LD_LIBRARY_PATH ${CMAKE_INSTALL_PREFIX}/lib)

# build the custom configuration file
configure_file(iscetest.py.in "${ISCE_BUILDPACKAGESDIR}/iscetest.py")

# recurse
add_subdirectory(extensions)
add_subdirectory(packages)
include(CTest)

# end of file
