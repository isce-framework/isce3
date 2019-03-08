
set(PACKAGENAME signal)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)
set(LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

set(ISCE_CUDA_${PACKAGENAME}_HEADERS
        gpuSignal.h
        gpuFilter.h
        gpuLooks.h
        gpuCrossMul.h)

set(LOCAL_SRCS
        gpuSignal.cu
        gpuFilter.cu
        gpuAzimuthFilter.cu
        gpuRangeFilter.cu
        gpuLooks.cu
        gpuCrossMul.cu)

# Append current path to every source file
set(ISCE_CUDA_${PACKAGENAME}_SRCS "")
foreach(CUFILE ${LOCAL_SRCS})
    list(APPEND ISCE_CUDA_${PACKAGENAME}_SRCS ${CMAKE_CURRENT_LIST_DIR}/${CUFILE})
endforeach()

# Install headers into build/include
set(${PACKAGENAME}_BUILD_HEADERS "")
foreach(HFILE ${ISCE_CUDA_${PACKAGENAME}_HEADERS})
    set(DESTFILE ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}/${HFILE})
    configure_file(${CMAKE_CURRENT_LIST_DIR}/${HFILE}
                   ${DESTFILE}
                   COPYONLY)
    list(APPEND ${PACKAGENAME}_BUILD_HEADERS ${DESTFILE})
endforeach()

# Add headers of dependencies
include_directories(${LCUDAISCE} PUBLIC
                    ${CEREAL_SRC_INCLUDEDIR}
                    ${ISCE_BUILDINCLUDEDIR}
                    ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}
                    ${GDAL_INCLUDE_DIR}
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Install headers into final installation directory
install(FILES ${${PACKAGENAME}_BUILD_HEADERS}
        DESTINATION ${ISCE_INCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}
        COMPONENT ISCE_CUDA)

# end of file
