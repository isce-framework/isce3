set(PACKAGENAME core)

set(HEADERS
        Common.h
        gpuBasis.h
        gpuComplex.h
        gpuEllipsoid.h
        gpuInterpolator.h
        gpuLinAlg.h
        gpuLUT1d.h
        gpuLUT2d.h
        gpuOrbit.h
        gpuPeg.h
        gpuPegtrans.h
        gpuPixel.h
        gpuPoly2d.h
        gpuProjections.h
        gpuStateVector.h )

set(SRCS
        gpuEllipsoid.cu
        gpuBilinearInterpolator.cu
        gpuBicubicInterpolator.cu
        gpuLinAlg.cu
        gpuLUT1d.cu
        gpuLUT2d.cu
        gpuOrbit.cu
        gpuPeg.cu
        gpuPegtrans.cu
        gpuPoly2d.cu
        gpuProjections.cu
        gpuSinc2dInterpolator.cu
        gpuSpline2dInterpolator.cu )

add_isce_libdir(core "${SRCS}" "${HEADERS}")
