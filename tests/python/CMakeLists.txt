# adjust the PYTHONPATH to point to the freshly built packages
set(PYTEST_PYTHONPATH "${ISCE_BUILDPACKAGESDIR}:$ENV{PYTHONPATH}"
  CACHE INTERNAL "python paths for pytest")

# build the custom configuration file
configure_file(iscetest.py.in "${ISCE_BUILDPACKAGESDIR}/iscetest.py")

# recurse
add_subdirectory(extensions)
add_subdirectory(packages)
include(CTest)

# end of file
