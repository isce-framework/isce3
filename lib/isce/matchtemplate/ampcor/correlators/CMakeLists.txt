###List the source files
set(SRCS
    c2r.cpp
    correlate.cpp
    detect.cpp
    maxcor.cpp
    migrate.cpp
    nudge.cpp
    offsets.cpp
    r2c.cpp
    refStats.cpp
    sat.cpp
    tgtStats.cpp)

set(HEADERS
    kernels.h
    Sequential.h
    Sequential.icc
    correlators.h)

add_isce_libdir(matchtemplate/ampcor/correlators "${SRCS}" "${HEADERS}")
