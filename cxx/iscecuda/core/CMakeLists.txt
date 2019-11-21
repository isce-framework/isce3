set(SOURCES
    Event.cu
    gpuBilinearInterpolator.cu
    gpuBicubicInterpolator.cu
    gpuLUT1d.cu
    gpuLUT2d.cu
    gpuPoly2d.cu
    gpuProjections.cu
    gpuSinc2dInterpolator.cu
    gpuSpline2dInterpolator.cu
    Orbit.cu
    Stream.cu
    )

set(HEADERS
    forward.h
    Event.h
    gpuBasis.h
    gpuInterpolator.h
    gpuLUT1d.h
    gpuLUT2d.h
    gpuPoly2d.h
    gpuProjections.h
    Orbit.h
    OrbitView.h
    OrbitView.icc
    Stream.h
    )

add_isce_libdir(core "${SOURCES}" "${HEADERS}")
