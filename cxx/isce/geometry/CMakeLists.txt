####List the source files
set(SRCS
    DEMInterpolator.cpp
    Geo2rdr.cpp
    geometry.cpp
    RTC.cpp
    Topo.cpp
    Geocode.cpp
    boundingbox.cpp)

#####Library headers
set(HEADERS
    forward.h
    DEMInterpolator.h
    Geo2rdr.h
    Geo2rdr.icc
    geometry.h
    RTC.h
    Serialization.h
    Topo.h
    Topo.icc
    TopoLayers.h
    Geocode.h
    Geocode.icc
    boundingbox.h
    Shapes.h)

add_isce_libdir(geometry "${SRCS}" "${HEADERS}")
