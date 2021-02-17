list(APPEND PYTEST_PYTHONPATH ${ISCE_BUILDPACKAGESDIR} $ENV{PYTHONPATH})
string(REPLACE ";" ":" PYTEST_PYTHONPATH "${PYTEST_PYTHONPATH}")

if(DEFINED ENV{DESTDIR})
    set(PYTEST_LD_LIBRARY_PATH "$ENV{DESTDIR}/${CMAKE_INSTALL_FULL_LIBDIR}:$ENV{LD_LIBRARY_PATH}")
else()
    set(PYTEST_LD_LIBRARY_PATH "${CMAKE_INSTALL_FULL_LIBDIR}:$ENV{LD_LIBRARY_PATH}")
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
