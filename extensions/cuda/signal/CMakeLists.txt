set(HEADERS
        gpuSignal.h
        gpuFilter.h
        gpuLooks.h
        gpuCrossMul.h)

set(SRCS
        gpuSignal.cu
        gpuFilter.cu
        gpuAzimuthFilter.cu
        gpuRangeFilter.cu
        gpuLooks.cu
        gpuCrossMul.cu)


add_isce_libdir(image "${SRCS}" "${HEADERS}")
