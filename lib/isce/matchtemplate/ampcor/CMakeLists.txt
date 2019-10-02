###List the source files
set(SRCS
    correlators/c2r.cpp
    correlators/correlate.cpp
    correlators/detect.cpp
    correlators/maxcor.cpp
    correlators/migrate.cpp
    correlators/nudge.cpp
    correlators/offsets.cpp
    correlators/r2c.cpp
    correlators/refStats.cpp
    correlators/sat.cpp
    correlators/tgtStats.cpp
    dom/Raster.cc
    dom/SLC.cc)

set(HEADERS
    correlators.h
    correlators/kernels.h
    correlators/Sequential.h
    correlators/Sequential.icc
    correlators/public.h
    dom.h
    dom/Raster.h
    dom/Raster.icc
    dom/SLC.h
    dom/SLC.icc
    dom/public.h)

add_isce_libdir(matchtemplate/ampcor "${SRCS}" "${HEADERS}")
