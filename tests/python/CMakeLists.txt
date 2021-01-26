# adjust the PYTHONPATH to point to the freshly built packages
if(DEFINED ENV{DESTDIR})
    set(PYTEST_PYTHONPATH $ENV{DESTDIR}/${ISCE_PACKAGESDIR_FULL})
else()
    set(PYTEST_PYTHONPATH ${ISCE_PACKAGESDIR_FULL})
endif()
list(APPEND PYTEST_PYTHONPATH ${ISCE_BUILDPACKAGESDIR} $ENV{PYTHONPATH})
string(REPLACE ";" ":" PYTEST_PYTHONPATH "${PYTEST_PYTHONPATH}")

if(DEFINED ENV{DESTDIR})
    set(PYTEST_LD_LIBRARY_PATH $ENV{DESTDIR}/${CMAKE_INSTALL_FULL_LIBDIR})
else()
    set(PYTEST_LD_LIBRARY_PATH ${CMAKE_INSTALL_FULL_LIBDIR})
endif()

# build the custom configuration file
configure_file(iscetest.py.in "${ISCE_BUILDPACKAGESDIR}/iscetest.py")

# TODO check if pytest is importable
# Disable cache in case source dir is read-only, or pytest will spew warnings
set(PYTEST_EXECUTABLE ${Python_EXECUTABLE} -m pytest -r apP -p no:cacheprovider)

# recurse
add_subdirectory(extensions)
add_subdirectory(packages)
include(CTest)
