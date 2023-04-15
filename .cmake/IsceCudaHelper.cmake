# Return a list of compute capability for each installed CUDA device
#
# args
# ----
# COMPUTE_CAPABILITIES
#   output list of device compute capabilities
function(detect_devices COMPUTE_CAPABILITIES)
    # try compiling and running a script to detect installed CUDA devices
    try_run(
        RUN_RESULT COMPILE_RESULT
        ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/.cmake/DetectDevices.cu
        RUN_OUTPUT_VARIABLE CC)

    # check that compile & run succeeded
    if(${COMPILE_RESULT} AND ${RUN_RESULT} EQUAL 0)

        # strip trailing whitespace
        string(STRIP "${CC}" CC)

        # warn if no CUDA devices were found
        if ("${CC}" STREQUAL "")
            message(WARNING
                "No installed CUDA devices found, "
                "falling back to default compilation options")
        endif()

        # convert to list
        string(REPLACE " " ";" CC ${CC})
        # remove duplicates
        list(REMOVE_DUPLICATES CC)

    else()
        # compile or run failed, warn and return empty list
        message(WARNING
            "Failed to detect installed CUDA device architecture(s), "
            "falling back to default compilation options")
        set(CC)
    endif()

    # set output variable
    set(${COMPUTE_CAPABILITIES} ${CC} PARENT_SCOPE)
endfunction()

# Generate nvcc compile options for designated CUDA device architecture(s)
# and append them to CMAKE_CUDA_FLAGS
#
# Target architecture can be any of:
#   - comma-separated list of compute capabilities (e.g. 3.5,5.0,5.2)
#   - "Auto" to detect installed CUDA devices and target those architectures
#   - "" (empty) to use default compilation options
#
# See
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation-generate-code
# for more details of specifying real & virtual architectures.
#
# args
# ----
# TARGET_ARCHS
#   target architectures
function(set_cuda_arch_flags TARGET_ARCHS)
    # use default compilation flags if input is empty
    if("${TARGET_ARCHS}" STREQUAL "")
        return()
    endif()

    if("${TARGET_ARCHS}" STREQUAL "Auto")
        # auto-detect installed CUDA device architectures
        detect_devices(TARGET_ARCHS)
    else()
        # assume we received comma-separated target archs
        # convert to list
        string(REPLACE "," ";" TARGET_ARCHS ${TARGET_ARCHS})
    endif()

    # print target architectures
    string(REPLACE ";" ", " TARGET_ARCHS_STR "${TARGET_ARCHS}")
    message(STATUS
        "Generating PTX and CUDA device binary code for "
        "the following architectures : " ${TARGET_ARCHS_STR})

    # build gencode flags for each target architecture
    set(FLAGS "")
    foreach(ARCH ${TARGET_ARCHS})
        # change {major}.{minor} to {major}{minor}
        string(REPLACE "." "" ARCH ${ARCH})
        # add gencode flag
        set(FLAGS "${FLAGS} -gencode=arch=compute_${ARCH},code=sm_${ARCH}")
    endforeach()
    # enable lambdas in device code
    set(FLAGS "${FLAGS} --extended-lambda")
    # strip leading whitespace
    string(STRIP "${FLAGS}" FLAGS)

    # append new flags to CMAKE_CUDA_FLAGS
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${FLAGS}" PARENT_SCOPE)
endfunction()
