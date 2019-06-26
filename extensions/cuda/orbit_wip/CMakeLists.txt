set(PACKAGENAME orbit_wip)

set(HEADERS
    Orbit.h
    )

set(SRCS
    Orbit.cu
    )

add_isce_libdir("${PACKAGENAME}" "${SRCS}" "${HEADERS}")
