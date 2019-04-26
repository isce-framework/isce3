####List the source files
set(SRCS
    IH5.cpp
    IH5Dataset.cpp
    Raster.cpp
    )

#####Library headers
set(HEADERS
    Constants.h
    IH5.h
    IH5.icc
    Raster.h
    Raster.icc
    Serialization.h
    IH5Dataset.h
)

add_isce_libdir(io "${SRCS}" "${HEADERS}")
