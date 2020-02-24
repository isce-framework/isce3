# Otherwise, try to use gtest from github
if(NOT GTEST_FOUND)
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/googletest-src)
        find_package(Git)
        if(NOT GIT_FOUND)
            message(FATAL_ERROR "need git or gtest installed")
        endif()

        execute_process(COMMAND ${GIT_EXECUTABLE} clone
                                --depth 1
                                https://github.com/google/googletest
                                googletest-src
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        RESULT_VARIABLE GIT_CLONE_RESULT)
        if(NOT GIT_CLONE_RESULT EQUAL "0")
            message(FATAL_ERROR "git clone failed with ${GIT_CLONE_RESULT}")
        endif()
    endif()
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                     ${CMAKE_CURRENT_BINARY_DIR}/googletest-bld)
endif()

# Otherwise, try to use pyre from github
if(NOT HAVE_PYRE)
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/pyre-src)
        find_package(Git)
        if(NOT GIT_FOUND)
            message(FATAL_ERROR "need git or gtest installed")
        endif()

        execute_process(COMMAND ${GIT_EXECUTABLE} clone
                                --depth 1
                                https://github.com/pyre/pyre
                                pyre-src
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        RESULT_VARIABLE GIT_CLONE_RESULT)
        if(NOT GIT_CLONE_RESULT EQUAL "0")
            message(FATAL_ERROR "git clone failed with ${GIT_CLONE_RESULT}")
        endif()
    endif()
    # disable pyre's test suite
    set(BUILD_TESTING OFF)
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/pyre-src
                     ${CMAKE_CURRENT_BINARY_DIR}/pyre-bld)
endif()

install(DIRECTORY cereal/include/cereal DESTINATION include
        FILES_MATCHING PATTERN *.hpp PATTERN *.h)
add_library(cereal INTERFACE)
add_library(Cereal::Cereal ALIAS cereal)
target_include_directories(cereal SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cereal/include>)

install(DIRECTORY eigen3/Eigen DESTINATION include)
add_library(eigen INTERFACE)
add_library(Eigen::Eigen ALIAS eigen)
target_include_directories(eigen SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/eigen3>)

add_subdirectory(pybind11)
