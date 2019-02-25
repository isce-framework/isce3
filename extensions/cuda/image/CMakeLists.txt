set(HEADERS
        gpuImageMode.h
        gpuResampSlc.h
        ResampSlc.h)

set(SRCS
        gpuResampSlc.cu
        ResampSlc.cpp)

# end of file
add_isce_libdir(image "${SRCS}" "${HEADERS}")
