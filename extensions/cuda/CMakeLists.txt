set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

set (LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set (LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
message(STATUS "Flags: ${CUDA_NVCC_FLAGS}")

set(ISCE_CUDA_HEADERS
        gpuEllipsoid.h
        gpuLinAlg.h
        gpuOrbit.h
        gpuPeg.h
        gpuPegtrans.h )

set (ISCE_CUDA_SRCS 
        gpuEllipsoid.cu
        gpuLinAlg.cu
        gpuOrbit.cu
        gpuPeg.cu
        gpuPegtrans.cu )

##Install headers into build/include
set(${PACKAGENAME}_BUILD_HEADERS "")
foreach(HFILE ${ISCE_CUDA_HEADERS})
    configure_file(${HFILE} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda/${HFILE} COPYONLY)
    list(APPEND ${PACKAGENAME_BUILD_HEADERS} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda/${HFILE})
endforeach()

##Ensure C++11 for CUDA
list(APPEND CUDA_NVCC_FLAGS "--std c++11")
message(STATUS "CUDA_INCLUDE:  ${CUDA_INCLUDE_DIRS}")

cuda_add_library(${LCUDAISCE} SHARED
        ${ISCE_CUDA_SRCS})
target_include_directories(${LCUDAISCE} PUBLIC ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda )

target_link_libraries(${LCUDAISCE} ${LISCE} ${CUDA_LIBRARIES})
set(TARGET ${LCUDAISCE} CMAKE_CXX_STANDARD 11)

install(FILES   ${${PACKAGENAME}_BUILD_HEADERS}
    DESTINATION ${ISCE_INCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda
    COMPONENT isce_cuda)

install (TARGETS    ${LCUDAISCE}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_cuda)
