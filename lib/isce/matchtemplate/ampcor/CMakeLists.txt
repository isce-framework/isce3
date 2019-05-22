###List the source files
set(SRCS
    correlators/Sequential.cc
    dom/Raster.cc
    dom/SLC.cc)

set(HEADERS
    correlators.h
    correlators/Correlator.h
    correlators/Correlator.icc
    correlators/Sequential.h
    correlators/Sequential.icc
    correlators/SumArea.h
    correlators/SumArea.icc
    correlators/public.h
    dom.h
    dom/Raster.h
    dom/Raster.icc
    dom/SLC.h
    dom/SLC.icc
    dom/public.h)

add_isce_libdir(matchtemplate/ampcor "${SRCS}" "${HEADERS}")
