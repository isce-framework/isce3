set(SOURCES
    BuildOrbit.cpp
    )

set(HEADERS
    BuildOrbit.h
    InterpolateOrbit.h
    InterpolateOrbit.icc
    )

add_isce_libdir(core/detail "${SOURCES}" "${HEADERS}")
