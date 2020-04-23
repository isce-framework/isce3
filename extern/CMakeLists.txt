if(NOT Cereal_FOUND)
    install(DIRECTORY cereal/include/cereal DESTINATION include
            FILES_MATCHING PATTERN *.hpp PATTERN *.h)
    add_library(cereal INTERFACE)
    target_include_directories(cereal SYSTEM INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cereal/include>
        $<INSTALL_INTERFACE:include>)
    install(TARGETS cereal EXPORT isce3-targets)
endif()
add_library(Cereal::Cereal ALIAS cereal)

add_subdirectory(pybind11)
