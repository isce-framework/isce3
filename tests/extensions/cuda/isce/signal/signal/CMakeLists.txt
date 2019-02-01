set(PACKAGENAME image)
string(TOLOWER ${PROJECT_NAME} LOCALPROJ)

set(LISCE isce.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})
set(LCUDAISCE iscecuda.${ISCE_VERSION_MAJOR}.${ISCE_VERSION_MINOR})

add_executable(gpuSignal gpuSignal.cpp)

target_link_libraries(gpuSignal PUBLIC ${LISCE} ${LCUDAISCE} gtest)

target_include_directories(gpuSignal PUBLIC
                           ${ISCE_BUILDINCLUDEDIR}
                           ${CEREAL_SRC_INCLUDEDIR}
                           ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME}
                           ${GDAL_INCLUDE_DIR}
                           ${PYRE_INCLUDE_DIR}
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           )

message(STATUS ${ISCE_BUILDINCLUDEDIR}/${LOCALPROJ}/cuda/${PACKAGENAME})
add_test(NAME gpuSignal COMMAND gpuSignal)

include(CTest)
