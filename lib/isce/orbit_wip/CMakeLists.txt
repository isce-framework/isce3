set(PACKAGENAME orbit_wip)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

set(CPPS
    Orbit.cpp
    )

set(HEADERS
    InterpOrbit.h
    InterpOrbit.icc
    Orbit.h
    )

add_isce_libdir("${PACKAGENAME}" "${CPPS}" "${HEADERS}")
