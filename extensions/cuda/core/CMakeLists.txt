set(PACKAGENAME core)

set(HEADERS
        gpuBasis.h
        gpuInterpolator.h
        gpuLUT1d.h
        gpuLUT2d.h
        gpuOrbit.h
        gpuPoly2d.h
        gpuProjections.h
        gpuStateVector.h )

set(SRCS
        gpuBilinearInterpolator.cu
        gpuBicubicInterpolator.cu
        gpuLUT1d.cu
        gpuLUT2d.cu
        gpuOrbit.cu
        gpuPoly2d.cu
        gpuProjections.cu
        gpuSinc2dInterpolator.cu
        gpuSpline2dInterpolator.cu )

add_isce_libdir(core "${SRCS}" "${HEADERS}")
