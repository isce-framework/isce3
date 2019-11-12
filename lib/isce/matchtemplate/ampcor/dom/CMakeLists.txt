###List the source files
set(SRCS
    Raster.cc
    SLC.cc)

set(HEADERS
    Raster.h
    Raster.icc
    SLC.h
    SLC.icc
    dom.h)

add_isce_libdir(matchtemplate/ampcor/dom "${SRCS}" "${HEADERS}")
