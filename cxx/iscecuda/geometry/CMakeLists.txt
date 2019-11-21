set(HEADERS
        forward.h
        gpuDEMInterpolator.h
        gpuGeometry.h
        gpuRTC.h
        gpuTopoLayers.h
        gpuGeo2rdr.h
        Topo.h
        Geo2rdr.h)

set(SRCS
        gpuDEMInterpolator.cu
        gpuGeometry.cu
        gpuRTC.cu
        gpuTopo.cu
        gpuTopoLayers.cpp
        gpuGeo2rdr.cu
        utilities.cu
        Topo.cpp
        Geo2rdr.cpp)

# end of file
add_isce_libdir(geometry "${SRCS}" "${HEADERS}")
