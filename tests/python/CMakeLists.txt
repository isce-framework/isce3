# adjust the PYTHONPATH to point to the freshly built packages
set(PYTEST_PYTHONPATH
    $ENV{DESTDIR}/${ISCE_PACKAGESDIR_FULL}:${ISCE_BUILDPACKAGESDIR}:$ENV{PYTHONPATH})

set(PYTEST_LD_LIBRARY_PATH $ENV{DESTDIR}/${CMAKE_INSTALL_FULL_LIBDIR})

# build the custom configuration file
configure_file(iscetest.py.in "${ISCE_BUILDPACKAGESDIR}/iscetest.py")

# TODO check if pytest is importable
# Disable cache in case source dir is read-only, or pytest will spew warnings
set(PYTEST_EXECUTABLE ${Python_EXECUTABLE} -m pytest -r apP -p no:cacheprovider)

# recurse
add_subdirectory(extensions)
add_subdirectory(packages)
include(CTest)

# end of file
