if(NOT GTest_FOUND)
    set(BUILD_GMOCK   OFF CACHE INTERNAL "")
    set(INSTALL_GTEST OFF CACHE INTERNAL "")
    add_subdirectory(googletest)
    set_target_properties(gtest gtest_main PROPERTIES EXCLUDE_FROM_ALL TRUE)
    add_library(GTest::gtest ALIAS gtest)
endif()

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
