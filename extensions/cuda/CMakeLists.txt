set(PACKAGENAME core)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

set (LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set (LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

set(ISCE_CUDA_HEADERS
        gpuEllipsoid.h
        gpuLinAlg.h
        gpuOrbit.h
        gpuPeg.h
        gpuPegtrans.h
	gpuPoly2d.h
	gpuProjections.h )

set (ISCE_CUDA_SRCS 
        gpuEllipsoid.cu
        gpuLinAlg.cu
        gpuOrbit.cu
        gpuPeg.cu
        gpuPegtrans.cu
	gpuPoly2d.cu
        gpuProjections.cu )

##Install headers into build/include
set(${PACKAGENAME}_BUILD_HEADERS "")
foreach(HFILE ${ISCE_CUDA_HEADERS})
    set(DESTFILE ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda/${HFILE})
    configure_file(${HFILE} ${DESTFILE} COPYONLY)
    list(APPEND ${PACKAGENAME}_BUILD_HEADERS ${DESTFILE})
endforeach()

add_library(${LCUDAISCE} SHARED
        ${ISCE_CUDA_SRCS})
target_include_directories(${LCUDAISCE} PUBLIC ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME} ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda ${GDAL_INCLUDE_DIR})

get_target_property(MYPROPS  ${LCUDAISCE} COMPILE_OPTIONS)
#message(STATUS "CUDA prop: ${MYPROPS}")
string(REPLACE "c++17" "c++11" MYPROPS ${MYPROPS})
#message(STATUS "CUDA post: ${MYPROPS}")
set_target_properties(${LCUDAISCE} PROPERTIES COMPILE_OPTIONS ${MYPROPS})
set_target_properties(${LCUDAISCE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${LCUDAISCE} ${LISCE})
#target_compile_features(${LCUDAISCE} PUBLIC cxx_std_11)

install(FILES   ${${PACKAGENAME}_BUILD_HEADERS}
    DESTINATION ${ISCE_INCLUDEDIR}/${LOCALPROJ}/${PACKAGENAME}/cuda
    COMPONENT isce_cuda)

install (TARGETS    ${LCUDAISCE}
        DESTINATION ${ISCE_LIBDIR}
        COMPONENT isce_cuda)
