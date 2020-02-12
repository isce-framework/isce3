add_subdirectory(googletest)
if(NOT HAVE_PYRE)
    # disable pyre's test suite
    set(BUILD_TESTING OFF)
    add_subdirectory(pyre)
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
